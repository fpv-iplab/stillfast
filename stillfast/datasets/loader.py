from fvcore.common.registry import Registry
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
import torch
from stillfast.datasets import *
from stillfast.datasets.samplers import GroupBatchSampler, DistributedGroupBatchSampler, DistributedWeightedSampler
from torch.utils.data import WeightedRandomSampler

from .build import build_dataset

def list_collate(batch):
    return zip(*batch)

def dict_collate(batch):
    return {k: [d[k] for d in batch] for k in batch[0]}

def get_collate(key):
    return dict_collate 

def construct_loader(cfg, split):
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_DEVICES)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = True
        group_batch_sampler = cfg.TRAIN.GROUP_BATCH_SAMPLER
    elif split in ["val"]:
        dataset_name = cfg.VAL.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.VAL.BATCH_SIZE / cfg.NUM_DEVICES)
        else:
            batch_size = cfg.VAL.BATCH_SIZE
        shuffle = False
        drop_last = False
        group_batch_sampler = cfg.VAL.GROUP_BATCH_SAMPLER
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_DEVICES)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False
        group_batch_sampler = cfg.TEST.GROUP_BATCH_SAMPLER

    weighted_sampler = split=='train' and cfg.TRAIN.WEIGHTED_SAMPLER

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    sampler = None
    batch_sampler = None

    if group_batch_sampler:
        if weighted_sampler:
            raise ValueError("Group batch sampler is not supported with weighted sampler")
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_DEVICES > 1:
            batch_sampler = DistributedGroupBatchSampler(dataset, batch_size, drop_last)
        else:
            # Create a sampler for multi-process training
            batch_sampler = GroupBatchSampler(dataset, batch_size, drop_last)
    else:
        # Create a sampler for multi-process training
        if hasattr(dataset, "sampler"):
            sampler = dataset.sampler
        elif cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_DEVICES > 1:
            if weighted_sampler:
                sampler = DistributedWeightedSampler(dataset)
            else:
                sampler = DistributedSampler(dataset)
        else:
            if weighted_sampler:
                sampler = WeightedRandomSampler(dataset.weights, len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1 if batch_sampler else batch_size,
        shuffle=False if sampler or batch_sampler else shuffle,
        batch_sampler=batch_sampler,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=False if batch_sampler else drop_last,
        collate_fn=get_collate(cfg.TASK),
    )
    return loader