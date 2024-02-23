
import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    # data in/out configs
    parser.add_argument("--data_dir", type=str, help="data directory")
    parser.add_argument("--metrics_dir", type=str, help="metrics data directory")
    parser.add_argument("--output_dir", type=str, help="model output directory")
    parser.add_argument("--checkpoint_path", default="", type=str, help="checkpoint_path")
    parser.add_argument("--log_dir", type=str, help="log directory")
    parser.add_argument(
        "--textemb_inference_model_dir",
        default="xlm-roberta-base",
        type=str,
        help="model for inferencing text imbedding in data loader",
    )
    parser.add_argument("--cache_dir", type=str, help="log directory")
    parser.add_argument("--eval_only", action="store_true", help="run model evaluation only")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="overwrite output directory")
    parser.add_argument("--logging_steps", default=100, type=int, help="save training state every n steps")
    parser.add_argument("--save_checkpoint_steps", default=1000, type=int, help="save model checkpoint every n steps")
    parser.add_argument("--max_val_steps", default=1000, type=int, help="perform validation of max_val_steps batches")

    
    # training configs

    parser.add_argument("--per_gpu_train_batch_size", default=128, type=int, help="training batch size")
    parser.add_argument("--val_batch_size", default=256, type=int, help="val batch size")

    parser.add_argument("--max_sequence_len", type=int, default=384, help="input vdom node max count")
    parser.add_argument("--num_classes", type=int, default=6, help="num_classes")
    parser.add_argument("--max_token_len", default=5, type=int, help="max token length per vdom node")

    parser.add_argument("--lr", type=float, default=6e-4, help="peak rate of lr")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    
    parser.add_argument("--epoch", type=int, default=30, help="epoch")
    parser.add_argument("--scheduler_type", type=str, default="cosine_with_warmup", help="See scheduler_factory.py for options")
    parser.add_argument("--warmup_factor", type=float, default=0.05, help="percentage of the training that will be LR warmup.")
    parser.add_argument("--freeze_text_encoder", action="store_true", help="freeze text encoder from training")
    parser.add_argument("--cuda", type=str, help="cuda available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'] or 'pytorch_native' for PyTorch AMP."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, type=float, help="gradient accumulate every n steps"
    )
    parser.add_argument("--load_optimizer", action="store_true", help="load optimizer checkpoint or not")
    parser.add_argument(
        "--train_dataloader_worker_count", default=16, type=int, help="worker count for dataloader"
    )
    parser.add_argument(
        "--val_dataloader_worker_count", default=1, type=int, help="worker count for dataloader"
    )
    
    # model configs
    parser.add_argument("--model_version", type=str, default='text_bert', help="model architecture, for ablations")

    parser.add_argument("--text_in_emb_dim", default=768, type=int, help="embedding_dim")
    parser.add_argument("--text_emb_dim", type=int, default=256, help="embedding_dim")

    parser.add_argument("--num_layers", type=int, default=3, help="num_layers")
    parser.add_argument("--text_encoder_num_hidden_layer", type=int, default=1, help="num_layers")
    parser.add_argument("--num_heads", type=int, default=8, help="num_heads")
    parser.add_argument("--disable_positional_encoding", action="store_true", help="disable pos encoder")

    # Distributed configs
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    return parser
