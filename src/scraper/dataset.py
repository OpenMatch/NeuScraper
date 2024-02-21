import os
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


class DistributedAccessDataset(IterableDataset):
    def __init__(self, records, fn):
        super().__init__()
        self.num_replicas = -1
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("Not running in distributed mode")

        self.records = records
        self.fn = fn

    def change_seed(self, seed):
        self.records.change_seed(seed)

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_worker = num_workers * self.num_replicas
        worker_no = self.rank * num_workers + worker_id

        for i, record in enumerate(self.records.get_dist_iter(worker_no, total_worker)):
            rows = self.fn(record, i)
            for rec in rows:
                yield rec


def all_gather_cpu(data, prefix="tmp", cache_path="cache/", cleanup=True, ranks=None):

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # serialized to a Tensor
    pathholder = "{0}_part_{1}"
    self_pkl_path = pathholder.format(prefix, str(rank))

    os.makedirs(cache_path, exist_ok=True)

    with open(os.path.join(cache_path, self_pkl_path), "wb") as handle:
        pickle.dump(data, handle, protocol=4)

    dist.barrier()

    data_list = []
    if ranks is None:
        ranks = list(range(world_size))
    if rank in ranks:
        for i in range(world_size):
            if i != rank:
                pkl_path = pathholder.format(prefix, str(i))
                with open(os.path.join(cache_path, pkl_path), "rb") as handle:
                    d = pickle.load(handle)
                data_list.append(d)
            else:
                data_list.append(data)

    dist.barrier()
    if cleanup:
        try:
            os.remove(self_pkl_path)
        except:
            # do nothing
            cleanup = False

    dist.barrier()
    return data_list