from torch.utils.data import DistributedSampler, Dataset, WeightedRandomSampler
from typing import Iterator, List, Optional, Union, Iterable
from operator import itemgetter
import numpy as np
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler
import torch.distributed as dist
import math
import torch

class GroupBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, drop_last: bool, indices=None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        if indices is None:
            indices = np.arange(len(dataset))
        else:
            indices = np.array(indices)
        self.groups = np.array(dataset.groups)[indices]
            
        self.samplers = []
        for group in np.unique(self.groups):
            idx = indices[np.where(np.array(self.groups) == group)[0]]
            self.samplers.append(BatchSampler(SubsetRandomSampler(idx), batch_size, drop_last))

    def __iter__(self) -> Iterable[List[int]]:
        batches = []
        for sampler in self.samplers:
            batches.extend(list(sampler))
        np.random.shuffle(batches)
        return iter(batches)
    
    def __len__(self) -> int:
        return np.sum([len(s) for s in self.samplers])

class DistributedGroupBatchSampler(GroupBatchSampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool) -> None:
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.num_samples = int(math.ceil(len(dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible
        indices = list(range(len(dataset)))
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        super().__init__(dataset=dataset, batch_size=batch_size, drop_last=drop_last, indices=indices)

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = dataset.weights
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        idx = torch.multinomial(torch.from_numpy(self.weights[indices]), self.num_samples, self.replacement)
        
        return iter([indices[i] for i in idx])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch