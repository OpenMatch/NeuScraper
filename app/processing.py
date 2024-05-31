import torch
import ujson
import torch.nn.functional as F


def wrapped_process_fn(args):
    def fn(line, i):
        return content_extraction_process_fn(line, i, args)

    return fn

def wrapped_eval_process_fn(args):
    def fn(line, i):
        return content_extraction_process_fn(line, i, args, eval_mode=True)

    return fn

def wrapped_commoncrawl_process_fn(args):
    def fn(line, i):
        return content_commoncrawl_process_fn(line, i, args, eval_mode=True)

    return fn

def parse_data_file_json(doc_json):
    """
    Extract Text token id, Offsets, Labels, Visual Features from JSON
    """
    data = None
    try:
        data = ujson.loads(doc_json)
    except:
        print(doc_json)

    return data

def generate_data_as_tensors(data, args, eval_mode=False):
    """
    given label_idx and tokenized_text, return text's hash_arr and one-hot encoding
    """
    y = data["Labels"]
    token_ids = data["TokenId"]
    num_nodes = len(y)

    # truncate
    if num_nodes > args.max_sequence_len:
        y = y[0 : args.max_sequence_len]
        token_ids = token_ids[0 : args.max_sequence_len]

    y = torch.tensor(y, dtype=torch.float).flatten()
    token_ids = torch.cat([torch.tensor(token_ids, dtype=torch.long)[:, :args.max_token_len - 1], 2 * torch.ones(len(token_ids), 1, dtype=torch.long)], axis=1).flatten()
    token_masks = torch.where(token_ids != 1, 1, 0)
    

    y = F.pad(y, (0, args.num_classes * args.max_sequence_len - y.shape[0]))
    token_ids = F.pad(token_ids, (0, args.max_token_len * args.max_sequence_len - token_ids.shape[0]))
    token_masks = F.pad(token_masks, (0, args.max_token_len * args.max_sequence_len - token_masks.shape[0]))

    tensors = [token_ids, token_masks, y]

    if eval_mode:
        tensors.append(data['Url'])
        tensors.append(data['NodeIds'])
    return [tensors]

def generate_commoncrawl_data_as_tensors(data, args, eval_mode=False):
    """
    given label_idx and tokenized_text, return text's hash_arr and one-hot encoding
    """
    token_ids = data["TokenId"]
    num_nodes = len(token_ids)

    # truncate
    if num_nodes > args.max_sequence_len:
        token_ids = token_ids[0 : args.max_sequence_len]

    token_ids = torch.cat([torch.tensor(token_ids, dtype=torch.long)[:, :args.max_token_len - 1], 2 * torch.ones(len(token_ids), 1, dtype=torch.long)], axis=1).flatten()
    token_masks = torch.where(token_ids != 1, 1, 0)
    
    token_ids = F.pad(token_ids, (0, args.max_token_len * args.max_sequence_len - token_ids.shape[0]))
    token_masks = F.pad(token_masks, (0, args.max_token_len * args.max_sequence_len - token_masks.shape[0]))

    tensors = [token_ids, token_masks]

    tensors.append(data['Url'])
    tensors.append(data['NodeIds'])
    return [tensors]


def content_extraction_process_fn(line, i, args, eval_mode=False):
    data = parse_data_file_json(line)
    tensors = generate_data_as_tensors(data, args, eval_mode)

    return tensors


def content_commoncrawl_process_fn(line, i, args, eval_mode=False):
    data = parse_data_file_json(line)
    tensors = generate_commoncrawl_data_as_tensors(data, args, eval_mode)

    return tensors


def content_extraction_collate_fn(batch):
    num_fields = len(batch[0])
    final_batch = []
    
    for i in range(num_fields):
        if not torch.is_tensor(batch[0][i]):
            field = [item[i] for item in batch]
        else:
            field = torch.stack([item[i] for item in batch]) 

        final_batch.append(field)

    return final_batch
