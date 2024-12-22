# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import torch
import random
from copy import deepcopy
from functools import reduce
from torch.utils.data import Sampler, ConcatDataset


def datainfo(set_indices, k):
    # print information about benchmarks
    print('')
    print(f'{" Benchmarks":20}|{" Count":16}')
    print(f'{"-" * 37}')
    for k0, v0 in set_indices.items():
        line = f' {k0:19}|'
        line += f' {len(v0):<15}'
        print(line)
        print(f'{"-" * 37}')
    line = f' {"ALL (" + k + ") ":19}|'
    line += f' {sum([len(x) for x in set_indices.values()]):<15}'
    print(line)
    print(f'{"-" * 37}')
    print('')


class BalanceSampler(Sampler):
    """
    Balance sampler for Datasets
    """
    def __init__(self, data_source: ConcatDataset, gpuid: int, gpus: int, trains: list, maxlen: list):

        super().__init__(data_source)

        if not isinstance(data_source, ConcatDataset):
            raise TypeError("data_source should be torch.utils.data.ConcatDataset")

        self.gpus = gpus
        self.gpuid = gpuid
        self.trains = trains
        self.maxlen = maxlen
        self.data_source = data_source
        self.set_class = list(set([x.__class__ for x in data_source.datasets]))
        self.set_indices, self.n_samples = self.make_indices()

    def __len__(self):
        return self.n_samples

    def make_indices(self):
        # isinstance(self.data_source.datasets[0], self.sets_class[0])
        set_indices = {x.__name__:[] for x in self.set_class}
        set_weights = {x.__name__:[] for x in self.set_class}

        pre = 0
        for cur, cls in zip(self.data_source.cumulative_sizes, self.data_source.datasets):
            indices = list(range(pre, cur))
            set_indices[cls.__class__.__name__] += indices
            set_weights[cls.__class__.__name__] += [1/(len(indices)**0.75+1e-8)] * len(indices)
            pre = cur

        if self.gpuid == 0: datainfo(set_indices, 'Before')

        # make all dataset same length
        for k, v in set_indices.items():
            seed = 3407 if ('EPOCH' not in os.environ) else int(os.environ['EPOCH'])
            num = len(v)
            maxlen = [ n for d, n in zip(self.trains, self.maxlen) if d in k][0]
            assert maxlen % self.gpus == 0
            indices = torch.tensor(v)
            if num < maxlen:
                sample_num = maxlen - num
                rand_tensor = torch.randint(low=0, high=len(indices), size=(sample_num,), generator=torch.manual_seed(seed+1))
                sample_indices = indices[rand_tensor]
                indices = torch.cat([sample_indices, indices])
            elif num > maxlen:
                if maxlen > 0:
                    sampler = torch.utils.data.WeightedRandomSampler(set_weights[k], num_samples=maxlen, replacement=False, generator=torch.manual_seed(seed+2))
                    indices = indices[list(sampler)]
            rand_tensor = torch.randperm(len(indices), generator=torch.manual_seed(seed+3))
            indices = indices[rand_tensor]
            set_indices[k] = indices.tolist()

        if self.gpuid == 0: datainfo(set_indices, 'After')

        for k, v in set_indices.items(): set_indices[k] = v[self.gpuid::self.gpus]

        n_samples = sum([len(x) for x in set_indices.values()])
        if 'EPOCH' not in os.environ: print(f'[GPU {self.gpuid}] Train Dataset Length after Sampler: {n_samples}')
        return set_indices, n_samples

    def __iter__(self):
        set_indices = deepcopy(self.set_indices)
        for v in set_indices.values(): random.shuffle(v)
        set_indices = reduce(lambda x, y: x + y, set_indices.values())
        indices = [None] * len(set_indices)
        n = sum(self.maxlen) //  min(self.maxlen)
        num = len(set_indices) // n
        for i in range(n): indices[i::n] = set_indices[i*num:(i+1)*num]
        return iter(indices)


class ValidSampler(Sampler):
    """
    Balance sampler for Datasets
    """
    def __init__(self, data_source: ConcatDataset, gpuid: int, gpus: int, maxlen: int):

        super().__init__(data_source)

        if not isinstance(data_source, ConcatDataset):
            raise TypeError("data_source should be torch.utils.data.ConcatDataset")

        self.gpus = gpus
        self.gpuid = gpuid
        self.maxlen = maxlen
        self.data_source = data_source
        self.set_class = list(set([x.__class__ for x in data_source.datasets]))
        self.set_indices, self.n_samples = self.make_indices()

    def __len__(self):
        return self.n_samples

    def make_indices(self):
        # isinstance(self.data_source.datasets[0], self.sets_class[0])
        set_indices = {x.__name__:[] for x in self.set_class}

        pre = 0
        for cur, cls in zip(self.data_source.cumulative_sizes, self.data_source.datasets):
            indices = list(range(pre, cur))
            set_indices[cls.__class__.__name__] += indices
            pre = cur

        # make all dataset same length
        seed = 3407
        maxlen = self.maxlen
        for k, v in set_indices.items():
            num = len(v)
            assert num >= maxlen
            indices = torch.tensor(v)
            if num > maxlen:
                rand_tensor = torch.randperm(len(indices), generator=torch.manual_seed(seed+2))
                indices = indices[rand_tensor]
                indices = indices[:maxlen]
            rand_tensor = torch.randperm(len(indices), generator=torch.manual_seed(seed+3))
            indices = indices[rand_tensor]
            set_indices[k] = indices.tolist()

        for k, v in set_indices.items(): set_indices[k] = v[self.gpuid::self.gpus]

        n_samples = sum([len(x) for x in set_indices.values()])
        if 'EPOCH' not in os.environ: print(f'[GPU {self.gpuid}] Valid Dataset Length after Sampler: {n_samples}')
        return set_indices, n_samples

    def __iter__(self):
        set_indices = deepcopy(self.set_indices)
        indices = reduce(lambda x, y: x + y, [set_indices[k] for k in sorted(list(set_indices.keys()))])
        return iter(indices)
