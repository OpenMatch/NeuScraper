import logging
import mmap
import os
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

logger = logging.getLogger(__name__)

historical_max_buffer = 0
torch.multiprocessing.set_sharing_strategy("file_system")

debug_done = set()

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized(
    ) or dist.get_rank() == 0


OFFSET_CACHE_DIR = str(Path.home()) + "/cache/"


class LineShuffler:
    def __init__(self,
                 filepath,
                 seed=-1,
                 encoding="utf-8",
                 offsetmap_cache=None,
                 magic_num=1):
        self.path = filepath
        self.encoding = encoding
        self.offsetmap_cache = offsetmap_cache
        if self.offsetmap_cache is None:
            file_dir = os.path.dirname(filepath)
            file_name = os.path.basename(filepath)
            self.offsetmap_cache = filepath + "_offsets"
            if not os.path.exists(self.offsetmap_cache) and not os.access(
                    file_dir, os.W_OK | os.X_OK):
                os.makedirs(OFFSET_CACHE_DIR, exist_ok=True)
                self.offsetmap_cache = OFFSET_CACHE_DIR + file_name + "_offsets"
        with open(self.path, "rb") as f:
            self.offset_map = self.gen_offset_map(f)
        self.total_number = len(self.offset_map)
        self.change_seed(seed, magic_num)

    def change_seed(self, seed, magic_num):
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.ix_array = self.ix_array[:(len(self.ix_array) // magic_num *
                                        magic_num)]
        self.seed = seed

    def open(self):
        self.f = open(self.path, "rb", buffering=0)
        self.mm = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)

    def close(self):
        self.mm.close()
        self.f.close()

    def get_dist_iter(self, worker_no, total_worker):
        start = len(self.ix_array) // total_worker * worker_no + min(
            len(self.ix_array) % total_worker, worker_no)
        end = len(self.ix_array) // total_worker * (worker_no + 1) + min(
            len(self.ix_array) % total_worker, worker_no + 1)
        worker_array = self.ix_array[start:end]

        for ix in worker_array:
            line = self.__getitem__(ix)
            yield line

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __iter__(self):
        for ix in self.ix_array:
            line = self.__getitem__(ix)
            yield line

    def __getitem__(self, key):
        offset = self.offset_map[key]
        self.mm.seek(offset)
        line = self.mm.readline().decode(self.encoding)
        return line

    def __len__(self):
        return self.total_number

    def gen_new_offset_map(self, f):
        offset_map = []
        offset = f.tell()
        for _ in tqdm(f):
            offset_map.append(offset)
            offset = f.tell()

        return offset_map

    def gen_offset_map(self, f):
        if os.path.exists(self.offsetmap_cache):
            with open(self.offsetmap_cache, "rb") as om:
                offset_map = pickle.load(om)
        else:
            offset_map = self.gen_new_offset_map(f)
            with open(self.offsetmap_cache, "wb") as om:
                pickle.dump(offset_map, om)
        return offset_map
