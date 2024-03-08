import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from model import ContentExtractionTextEncoder
from arguments import create_parser
from processing import wrapped_eval_process_fn, content_extraction_collate_fn
import pandas as pd
import os




class SamplesDataset(IterableDataset):
    def __init__(self, data_path, fn):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(line)
        self.fn = fn

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for row in self.data:
            x = self.fn(row, 0)
            for rec in x:
                yield rec

def pad_list(x, max_len):
    list_len = len(x)
    x.extend([0] * (max_len - list_len))
    return x

class ContentExtractionDeepModel(nn.Module):
    def __init__(self, args):
        # TODO: Add config file
        super().__init__()
        self.model = model = ContentExtractionTextEncoder(args)
        self._load_model(args.model_path, model, args.device)
        model.to(args.device)

    def forward(self, x):
        return self.model(x)

    def _load_model(self, checkpoint_path, model, device):
        checkpoint_state_dict = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint_state_dict["model_state_dict"]
        model_dict = self.model.state_dict()

        pretrained_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k in model_dict and (v.size() == model_dict[k].size()):
                pretrained_dict[k] = v
            else:
                if k in model_dict:
                    print("error load checkpoint {0} {1} {2}".format(k, v.size(), model_dict[k].size()))
                else:
                    print("error load checkpoint not in dict : "+k)
                return

        for k, v in model_dict.items():
            if k in model_dict and (v.size() == model_dict[k].size()):
                continue
            if "module."+k in state_dict and (v.size() == model_dict["module."+k].size()):
                continue
            print("error load checkpoint: model {0} not in checkpoint".format(k))
            return

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict) 

    
def eval_on_leaderboard_set_vectorized(args, model):
    thresholds = [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    data_path = args.data_path
    data_process_fn = wrapped_eval_process_fn(args)
    dataset = SamplesDataset(data_path, data_process_fn)
    dataloader = DataLoader(
            dataset, batch_size=args.val_batch_size, num_workers=0, collate_fn=content_extraction_collate_fn, drop_last=False
        )

    predicted_nodes = {}
    added_urls = set()
    tasks = ['Primary', 'Heading', 'Title', 'Paragraph', 'Table', 'List']
    
    for task in tasks:
        predicted_nodes[task] = {}
        for thr in thresholds:
            predicted_nodes[task][thr]={}
    
    model.eval()

    for val_step, batch in enumerate(dataloader):
        model.eval()

        padded_list = [pad_list(x, args.max_sequence_len) for x in batch[4]]
        node_ids_batch = torch.tensor(padded_list).to(args.device)
        urls = batch[3]
        batch = batch[:2]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            output = model(batch)
        
        for thr in thresholds:
            predictions = (output > thr)
            for idx, task in enumerate(tasks):
                task_predictions = predictions[:, :, idx].squeeze()
                pred_nodes = torch.where(task_predictions, node_ids_batch, 0).cpu().numpy()
                for batch_idx, doc in enumerate(pred_nodes): # each element of the batch
                    elements = set(doc)

                    if 0 in elements:
                        elements.remove(0)
                    curr_url = urls[batch_idx]

                    if len(curr_url)==0:
                        continue
                    else:
                        curr_url = curr_url[0]

                    if curr_url not in predicted_nodes[task][thr]:
                        predicted_nodes[task][thr][curr_url] = set()
                    predicted_nodes[task][thr][curr_url].update(elements)

    return predicted_nodes

def save_predictions(pred_nodes, args):
    thresholds = [0.9,0.25,0.25,0.1,0.1,0.1]
    tasks = ['Primary', 'Heading', 'Title', 'Paragraph', 'Table', 'List']
    
    for idx, task in enumerate(tasks):
        if task == "Primary":
            rows = []
            task_thr = thresholds[idx]
            task_pred_nodes = pred_nodes[task][task_thr]
            for url, nodes in task_pred_nodes.items():
                rows.extend([(url, int(node), task) for node in nodes])
            res_df = pd.DataFrame(rows, columns=['Url', 'TextNodeId', 'Task'])
    res_df.to_csv('temp/inference_test.tsv', sep='\t', encoding='utf-8', index=False)



if __name__ == "__main__":
    parser = create_parser()

    parser.add_argument("--model_path", type=str, help="model directory")
    parser.add_argument("--data_path",type=str, help="data path")

    if not os.path.exists('temp/'):
        os.makedirs('temp/')

    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = 0

    model = ContentExtractionDeepModel(args)

    pred_nodes = eval_on_leaderboard_set_vectorized(args, model)
    save_predictions(pred_nodes, args)