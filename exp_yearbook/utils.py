import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from flaml.group import set_test_group
from flaml import AutoML, CFO, tune
from collections import defaultdict
import pandas as pd
import argparse
import pickle
import itertools
import sys
import os
import numpy as np
from ray import tune as raytune
import sys
from flaml.data import load_openml_dataset
import time
import random
import arff
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _InfiniteSampler(torch.utils.data.Sampler):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, subsetids, collate_fn=None):
        super().__init__()
        if subsetids != None:
            using_dataset = Subset(dataset, subsetids)
            sampler = torch.utils.data.RandomSampler(using_dataset, replacement=True)
            # sampler = torch.utils.data.SubsetRandomSampler(dataset, subsetids)
        else:
            using_dataset = dataset
            sampler = torch.utils.data.RandomSampler(using_dataset, replacement=True)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=min(batch_size, len(using_dataset)),
            drop_last=True)
        self._infinite_iterator = iter(DataLoader(
            using_dataset,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=collate_fn,
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


def mixup_data(x, y, mix_alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if mix_alpha > 0:
        lam = np.random.beta(mix_alpha, mix_alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def collate_fn_mimic(batch):
    codes = [item[0][0] for item in batch]
    types = [item[0][1] for item in batch]
    target = [item[1] for item in batch]
    if len(batch[0]) == 2:
        return [(codes, types), target]
    else:
        groupid = torch.cat([item[2] for item in batch], dim=0).unsqueeze(1)
        return [(codes, types), target, groupid]


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""

    def __init__(self, dataset, batch_size, subsetids, collate_fn=None):
        super().__init__()
        num_workers = 0
        if subsetids != None:
            using_dataset = Subset(dataset, subsetids)
            tem_sampler = torch.utils.data.RandomSampler(using_dataset, replacement=False)
        else:
            using_dataset = dataset
            tem_sampler = torch.utils.data.RandomSampler(using_dataset, replacement=False)
        batch_sampler = torch.utils.data.BatchSampler(
            tem_sampler,
            batch_size=min(batch_size, len(using_dataset)),
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            using_dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=collate_fn
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
