import os, sys

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dir_path)

from dataset import DistributedAccessDataset, all_gather_cpu
from dataset_utils import LineShuffler
from prettytable import PrettyTable

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch import optim
import torch.distributed as dist
#from apex import amp

from transformers import XLMRobertaModel, XLMRobertaTokenizer

import math

import json
from tqdm import tqdm
import logging
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# BlingSD Model imports
from model import *
from schedulers import LearningRateScheduler

from processing import wrapped_process_fn, content_extraction_collate_fn
from metrics import *
from arguments import create_parser

curr_dir = os.path.dirname( __file__ )
metric_dir = os.path.join(curr_dir, '..', 'metric')
sys.path.append(metric_dir)

logger = logging.getLogger(__name__)


def setup_distributed(args, logger):
    # Setup CUDA devices
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )

    if args.local_rank != -1:
        print("set dist: gpu="+str(args.n_gpu)+" rank="+str(torch.distributed.get_rank()))


def load_model_and_setup_training(args, model):
    print("*** enter load_model_and_setup_training *** ")
    logger.info("***enter load_model_and_setup_training ***")
    # load model
    model, optimizer_state_dict, epoch, last_global_step, scheduler_state_dict = load_model(args, model)
    model.to(args.device)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # layerwise optimization for lamb
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
                 weight_decay=1e-2)

    if args.fp16 and args.fp16_opt_level != "pytorch_native":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    return model, optimizer, epoch, last_global_step, scheduler_state_dict


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def save_checkpoint(args, epoch, global_step, model, tokenizer, weight_map, result, optimizer, scheduler):
    if global_step < 0:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.module if hasattr(model, "module") else model

    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    torch.save(checkpoint_state_dict, os.path.join(output_dir, "training_state_checkpoint_{}.tar".format(global_step)))
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    with open(os.path.join(output_dir, "weight_mapping.json"), "w") as f:
        json.dump(weight_map, f)

    logger.info("Saving model checkpoint to %s", output_dir)


def load_model(args, model):
    logger.info(" *** load_model to ")
    if not os.path.exists(args.checkpoint_path):
        logger.info(" *** load_model to init")
        initialize_model_weights(args, model)
        epoch = 0
        last_global_step = 0
        optimizer_state_dict = None
        scheduler_state_dict = None
    else:
        model, optimizer_state_dict, epoch, last_global_step, scheduler_state_dict = load_checkpoint(args, model)

    if args.freeze_text_encoder:
        logger.info("Freeze Text Encoder? " + str(args.freeze_text_encoder))
        for param in model.text_roberta.parameters():
            param.requires_grad = False

    return model, optimizer_state_dict, epoch, last_global_step, scheduler_state_dict


def load_checkpoint(args, model):

    print(" **** load_checkpoint ****")
    logger.info("*** load_checkpoint *** "+args.checkpoint_path)
    checkpoint_state_dict = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))

    logger.info("***enter load_checkpoint *** "+str(len(checkpoint_state_dict)))
    # from train import model
    state_dict = checkpoint_state_dict["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and (v.size() == model_dict[k].size()):
            pretrained_dict[k] = v

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    optimizer_state_dict = None
    scheduler_state_dict = None
    if args.load_optimizer:
        optimizer_state_dict = checkpoint_state_dict["optimizer_state_dict"]
        epoch = checkpoint_state_dict["epoch"]
        last_global_step = checkpoint_state_dict["last_global_step"]
        scheduler_state_dict = checkpoint_state_dict["scheduler_state_dict"]
    else:
        epoch = 0
        last_global_step = 0

    del checkpoint_state_dict
    return model, optimizer_state_dict, epoch + 1, last_global_step, scheduler_state_dict


def initialize_model_weights(args, model):
    pretrain_model = XLMRobertaModel.from_pretrained(args.textemb_inference_model_dir)

    model.text_roberta.pooler = pretrain_model.pooler
    model.text_roberta.embeddings = pretrain_model.embeddings

    if args.text_encoder_num_hidden_layer == 12:
        model.encoder = pretrain_model.encoder
    else:
        layers_replace_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        for layer_i in layers_replace_map:
            if layer_i < args.text_encoder_num_hidden_layer:
                model.text_roberta.encoder.layer[layer_i] = pretrain_model.encoder.layer[layer_i]


def list_pt_files(base_dir):
    d = []
    for (dir_path, dir_names, file_names) in os.walk(base_dir):
        d.extend(os.path.join(dir_path, f) for f in file_names if (("tsv" in f or "json" in f) and not "offset" in f))
    return d


class DirectoryDataIteratorW_OffSetMap:
    def __init__(self, dir_path, encoding="utf-8", seed=-1, no_shuffle=False):
        self.files = list_pt_files(dir_path)
        if len(self.files) == 0:
            raise ValueError("Data directory contains 0 files")

        self.no_shuffle = no_shuffle
        self.state_cnt = 0
        self.encoding = encoding
        self.seed=seed

    def get_dist_iter(self, worker_no, total_worker):
        for file in self.files:
            seed = self._get_seed()
            with LineShuffler(file, seed=seed, encoding=self.encoding) as f:
                for i, record in enumerate(f.get_dist_iter(worker_no, total_worker)):
                    yield record
                      
    def __iter__(self):
        for file in self.files:
            seed = self._get_seed()
            with LineShuffler(file, seed=seed, encoding=self.encoding) as f:
                for i, record in enumerate(f.__iter__()):
                    yield record

    def __len__(self):
        total_rows = 0
        for file in self.files:
            seed = self._get_seed()
            with LineShuffler(file, seed=seed, encoding=self.encoding) as f:
                total_rows = total_rows + f.__len__()
        return total_rows
    
    def _get_seed(self):
        if self.no_shuffle:
            return -1
        self.state_cnt += 1
        return np.random.RandomState(self.seed + self.state_cnt).randint(10000000)

class DirectAccessDataset(IterableDataset):
    def __init__(self, records, fn):
        super().__init__()

        self.num_replicas = 1
        self.rank = 0
        self.records = records
        self.fn = fn

    def change_seed(self, seed):
        self.records.change_seed(seed)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        for i, record in enumerate(self.records.__iter__()):
            rows = self.fn(record, i)
            for rec in rows:
                yield rec


class ContentExtractionTrainer:
    def __init__(self, args, model, tokenizer, loss_calculator, data_process_fn):
        self.model, self.optimizer, self.epoch, self.global_step, self.scheduler_state_dict = load_model_and_setup_training(
            args, model
        )
        self.output_dir = args.output_dir
        self.loss_calculator = loss_calculator
        self.process_fn = data_process_fn
        self.tokenizer = tokenizer
        self.tr_loss = 0.0
        self.logging_loss = 0.0
        self.logging_loss_scalar = 0.0
        self.val_metrics_result = {}
        self.val_tr_loss = 0.0
        self.tb_writer = None
        self.train_data_path = os.path.join(os.path.abspath(args.data_dir), "train/")
        self.val_data_path = os.path.join(os.path.abspath(args.data_dir), "val/")
        self.train_dataloader_worker_count = args.train_dataloader_worker_count
        self.val_dataloader_worker_count = args.val_dataloader_worker_count

        if is_first_worker():
            self.tb_writer = SummaryWriter(log_dir=args.log_dir)

    def train(self, args):

        if is_first_worker():
            self.count_parameters()

        logger.info("Train Data Path: " + self.train_data_path)
        logger.info("Validation Data Path: " + self.val_data_path)

        foundNan = False

        logger.info("init data loader")
        train_datapath_iterator = DirectoryDataIteratorW_OffSetMap(self.train_data_path, seed=42)
        train_sds = DistributedAccessDataset(train_datapath_iterator, self.process_fn)
        train_dataloader = DataLoader(
            train_sds, batch_size=args.train_batch_size, num_workers= self.train_dataloader_worker_count, collate_fn=content_extraction_collate_fn)
        logger.info("Done creating data loader")
        total_steps = (len(train_dataloader) * args.epoch ) // (args.gradient_accumulation_steps)
        total_steps_per_process_per_epoch = len(train_dataloader) // (args.gradient_accumulation_steps * args.world_size)
        # Enforce the epoch to finish after last full checkpoint and also avoid any tail-related issues
        if args.save_checkpoint_steps > total_steps_per_process_per_epoch:
            args.save_checkpoint_steps = total_steps_per_process_per_epoch
        total_steps_per_process_per_epoch = (total_steps_per_process_per_epoch // args.save_checkpoint_steps) * args.save_checkpoint_steps
        total_steps_per_process = total_steps_per_process_per_epoch * args.epoch
        logger.info("Dataset Size: " + str(len(train_dataloader)))
        logger.info("Total Training Steps: " + str(total_steps_per_process))
        logger.info("Total Training Steps Per Epoch: " + str(total_steps_per_process_per_epoch))
        

        self.scheduler_factory = LearningRateScheduler(
            self.optimizer,
            num_warmup_steps=int(args.warmup_factor * total_steps_per_process), 
            num_training_steps=total_steps_per_process,
            scheduler_type=args.scheduler_type,
            lr = args.lr,
            epoch_steps = total_steps_per_process_per_epoch
        )

        self.scheduler = self.scheduler_factory.get_scheduler()

        if self.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(self.scheduler_state_dict)
            logger.info("Scheduler loaded from checkpoint.")
            logger.info(self.scheduler_state_dict)

        if args.fp16 and args.fp16_opt_level == "pytorch_native":
            self.scaler = GradScaler()

        for epoch in range(self.epoch, self.epoch + args.epoch):
            logger.info("\n-----------------------------------------------")
            logger.info("start epoch " + str(epoch) + " Worker "+str(dist.get_rank())+" nan:"+("yes" if foundNan else "no"))
            if foundNan:
                break
            # Data loader
            logger.info("Starting epoch " + str(epoch))
            # TODO RECORD TIMES
            for step, batch in tqdm(enumerate(train_dataloader)):
                if args.max_steps != -1 and 0 < args.max_steps < self.global_step:
                    logger.info("Reach max step, break training")
                    break
                if step >= total_steps_per_process_per_epoch:
                    break
                # Train model
                self.train_step(args, batch, step, epoch)

                if not math.isfinite(self.tr_loss):
                    logger.warning("Invalid loss" + str(self.tr_loss) + "Worker no. " + str(dist.get_rank()))
                    foundNan = True
                    break
                    
                if (self.global_step % args.save_checkpoint_steps) == 0:
                    # evaluate validation set and add metrics result to logs
                    self.validation(args)

                    if is_first_worker():
                        save_checkpoint(
                            args,
                            epoch,
                            self.global_step,
                            self.model,
                            self.tokenizer,
                            self.loss_calculator.get_weight_map(),
                            self.val_metrics_result,
                            self.optimizer,
                            self.scheduler,
                        )
                dist.barrier()          

            logger.info("Completed epoch " + str(epoch) + " Worker "+str(dist.get_rank())+" nan:"+("yes" if foundNan else "no"))

        if is_first_worker():
            self.tb_writer.close()

        return self.global_step, self.tr_loss, self.val_metrics_result

    def train_step(self, args, batch, step, epoch):
        self.model.train()

        if args.cuda:
            batch = tuple(t.to(args.device) for t in batch)

        labels = batch[2]
        batch = batch[:2]

        if args.fp16 and args.fp16_opt_level == "pytorch_native":
            with autocast():
                output = self.model(batch)
        else:
            output = self.model(batch)

        labels = labels.to(self.model.device).view(-1, args.max_sequence_len, args.num_classes)
        loss, class_loss = self.loss_calculator.weighted_crossentropy(labels, output)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16 and args.fp16_opt_level == "pytorch_native":
            self.scaler.scale(loss).backward()
        elif args.fp16: # AMP mixed precision
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if (step + 1) % args.gradient_accumulation_steps == 0:
                loss.backward()
            else:
                with self.model.no_sync():
                    loss.backward()

        self.tr_loss += loss.item()

        if not math.isfinite(loss.item()):
            logger.info("Step that has nan loss: " + str(step))

        if (self.global_step % args.logging_steps) == 0 and self.global_step != 0:
            #logger.info("Reach logging step")
            #logger.info("Loss: " + str(self.tr_loss))

            # tensorboard logs
            logs = {}

            loss_scalar = (self.tr_loss - self.logging_loss) / args.logging_steps
            learning_rate_scalar = self.scheduler.get_last_lr()[0]
            logs["learning_rate"] = learning_rate_scalar
            logs["loss_scalar"] = loss_scalar
            logs["loss"] = self.tr_loss
            self.logging_loss = self.tr_loss
            self.logging_loss_scalar = loss_scalar
            logger.info("Loss scalar: " + str(loss_scalar))

            if is_first_worker():
                for key, value in logs.items():
                    if isinstance(value, dict):
                        self.tb_writer.add_scalars(key, value, self.global_step)
                    else:
                        self.tb_writer.add_scalar(key, value, self.global_step)
                for name, weight in self.model.named_parameters():
                    if weight is None or weight.grad is None:
                        continue

                self.tb_writer.add_scalar("epoch", epoch, self.global_step)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16 and args.fp16_opt_level == "pytorch_native":
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler_factory.step()
                self.global_step += 1
            else:
                self.optimizer.step()
                self.scheduler_factory.step()
                self.model.zero_grad()
                self.global_step += 1

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def validation(self, args):
        val_metrics_results_list = []
        self.val_tr_loss = 0.0
        total_val_steps = 0.0

        val_datapath_iterator = DirectoryDataIteratorW_OffSetMap(self.val_data_path)
        val_sds = DistributedAccessDataset(val_datapath_iterator, self.process_fn)
        dataloader = DataLoader(
            val_sds, batch_size=args.val_batch_size, num_workers= self.val_dataloader_worker_count, collate_fn=content_extraction_collate_fn)

        logger.info("Running Validation")

        for val_step, batch in tqdm(enumerate(dataloader)):
            if total_val_steps > args.max_val_steps:
                break

            labels = batch[2]
            batch = batch[:2]
            self.model.eval()
            if args.cuda:
                batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                labels.to(self.model.device)
                labels = labels.to(self.model.device).view(-1, args.max_sequence_len, args.num_classes)
                output = self.model(batch)
                loss, class_loss = self.loss_calculator.weighted_crossentropy(labels, output)
                pr_result = self.loss_calculator.precision_recall_evaluation(labels, output)
                
                loss = loss.mean()
                if not math.isfinite(loss.item()):
                    logger.info("\nInvalid loss" + str(loss.item()) + "Worker no. " + str(dist.get_rank()))
                    logger.info("\nStep that has nan loss: " + str(val_step) + "Worker no. " + str(dist.get_rank()))
                    logger.info("\nper class loss: " + "Worker no. " + str(dist.get_rank()))
                    logger.info(class_loss)
                    continue

            val_metrics_results_list.append(pr_result)
            self.val_tr_loss += loss.item()
            total_val_steps += 1.0

        logger.info("Gathering results from all workers...")
        dist.barrier()
        val_metrics_results_cross_worker = all_gather_cpu(val_metrics_results_list, cache_path=args.cache_dir)
        if is_first_worker():
            logs = {}
            validation_loss = self.val_tr_loss / total_val_steps
            logs["validation_loss"] = validation_loss
            logs["loss_gap"] = self.logging_loss_scalar - validation_loss
            logger.info("Validation Loss: " + str(self.val_tr_loss))

            for key, value in logs.items():
                if isinstance(value, dict):
                    self.tb_writer.add_scalars(key, value, self.global_step)
                else:
                    self.tb_writer.add_scalar(key, value, self.global_step)


    
    def _pad_list(self, x, max_len):
        list_len = len(x)
        x.extend([0] * (max_len - list_len))
        return x


def main():
    parser = create_parser()
    args = parser.parse_args()

    setup_distributed(args, logger)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.eval_only
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    if is_first_worker():
        os.makedirs(args.output_dir, exist_ok=True)
    args.cache_dir=args.output_dir + "/cache"
    args.log_dir=args.output_dir + "/log"
    logger.info(args)

    loss_calculator = ContentExtractionLoss()
    model = ContentExtractionTextEncoder(args)

    tokenizer = XLMRobertaTokenizer.from_pretrained(
        args.textemb_inference_model_dir,
        do_lower_case=True,
        cache_dir=None,
    )
    data_process_fn = wrapped_process_fn(args)

    trainer = ContentExtractionTrainer(args, model, tokenizer, loss_calculator, data_process_fn)
    if not args.eval_only:
        global_step, tr_loss, metrics_result = trainer.train(args)

        if is_first_worker():
            logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            save_checkpoint(
                args, -1, -1, model, tokenizer, loss_calculator.get_weight_map(), metrics_result, trainer.optimizer, trainer.scheduler
            )

    dist.barrier()

    trainer.validation(args)
    logger.info("Exiting...")

    return


if __name__ == "__main__":
    main()
    os._exit(0)

        
