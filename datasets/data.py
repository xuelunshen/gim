# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import torch
import pytorch_lightning as pl

from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, ConcatDataset

from tools.misc import tqdm_joblib
from datasets.augment import build_augmentor
from datasets.sampler import BalanceSampler, ValidSampler
from .megadepth.megadepth import MegaDepthDataset
from .scannet.scannet import ScanNetDataset
from .walk.walk import WALKDataset

Benchmarks = dict(
    MegaDepth     = MegaDepthDataset,
    ScanNet       = ScanNetDataset,
    WALK          = WALKDataset,
)


class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined 
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, dcfg):
        """
        
        Args:
            args: (ArgumentParser) The only useful args is args.trains and args.valids
                    each one is a list, which contain like [PhotoTourism, MegaDepth,...]
                    We should traverse each item in args.trains and args.valids to build
                    self.train_datasets and self.valid_datasets
            dcfg: (yacs) It contain all configs for each benchmark in args.trains and
                    args.valids
        """
        super().__init__()

        self.args = args
        self.dcfg = dcfg
        self.train_loader_params = {'batch_size': args.batch_size,
                                    # 'shuffle': True,
                                    'num_workers': args.threads,
                                    'pin_memory': True,
                                    'drop_last': False}
        self.valid_loader_params = {'batch_size': args.valid_batch_size,
                                    'shuffle': False,
                                    'num_workers': args.threads,
                                    'pin_memory': True,
                                    'drop_last': False}
        self.tests_loader_params = {'batch_size': args.batch_size,
                                    'shuffle': False,
                                    'num_workers': args.threads,
                                    'pin_memory': True,
                                    'drop_last': False}

    def setup(self, stage=None):
        """ 
        Setup train/valid/test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        self.gpus = self.trainer.gpus * self.trainer.num_nodes
        self.gpuid = self.trainer.global_rank

        self.train_datasets = None
        self.valid_datasets = None
        self.tests_datasets = None

        # TRAIN
        if stage == 'fit':
            train_datasets = []
            for benchmark in self.args.trains:
                dcfg = self.dcfg.get(benchmark, None)
                assert dcfg is not None, "Training dcfg is None"

                datasets = self._setup_dataset(
                    benchmark=benchmark,
                    data_root=dcfg.DATASET.TRAIN.DATA_ROOT,
                    npz_root=dcfg.DATASET.TRAIN.NPZ_ROOT,
                    scene_list_path=dcfg.DATASET.TRAIN.LIST_PATH,
                    df=self.dcfg.DF,
                    padding=dcfg.DATASET.TRAIN.PADDING,
                    min_overlap_score=dcfg.DATASET.TRAIN.MIN_OVERLAP_SCORE,
                    max_overlap_score=dcfg.DATASET.TRAIN.MAX_OVERLAP_SCORE,
                    max_resize=self.args.img_size,
                    augment_fn=build_augmentor(dcfg.DATASET.TRAIN.AUGMENTATION_TYPE),
                    max_samples=dcfg.DATASET.TRAIN.MAX_SAMPLES,
                    mode='train',
                    njobs=dcfg.NJOBS,
                    cfg=dcfg.DATASET.TRAIN,
                )
                train_datasets += datasets
            self.train_datasets = ConcatDataset(train_datasets)
            os.environ['TOTAL_TRAIN_SAMPLES'] = str(len(self.train_datasets))

            # VALID
            valid_datasets = []
            for benchmark in self.args.valids:
                dcfg = self.dcfg.get(benchmark, None)
                assert dcfg is not None, "Validing dcfg is None"

                datasets = self._setup_dataset(
                    benchmark=benchmark,
                    data_root=dcfg.DATASET.VALID.DATA_ROOT,
                    npz_root=dcfg.DATASET.VALID.NPZ_ROOT,
                    scene_list_path=dcfg.DATASET.VALID.LIST_PATH,
                    df=self.dcfg.DF,
                    padding=dcfg.DATASET.VALID.PADDING,
                    min_overlap_score=dcfg.DATASET.VALID.MIN_OVERLAP_SCORE,
                    max_overlap_score=dcfg.DATASET.VALID.MAX_OVERLAP_SCORE,
                    max_resize=[1280, 1280],
                    augment_fn=build_augmentor(dcfg.DATASET.VALID.AUGMENTATION_TYPE),
                    max_samples=dcfg.DATASET.VALID.MAX_SAMPLES,
                    mode='valid',
                    njobs=dcfg.NJOBS,
                    cfg=dcfg.DATASET.VALID,
                )
                valid_datasets += datasets
            self.valid_datasets = ConcatDataset(valid_datasets)
            os.environ['TOTAL_VALID_SAMPLES'] = str(len(self.valid_datasets))

        # TEST
        if stage == 'test':
            tests_datasets = []
            for benchmark in [self.args.tests]:
                dcfg = self.dcfg.get(benchmark, None)
                assert dcfg is not None, "Validing dcfg is None"

                datasets = self._setup_dataset(
                    benchmark=benchmark,
                    data_root=dcfg.DATASET.TESTS.DATA_ROOT,
                    npz_root=dcfg.DATASET.TESTS.NPZ_ROOT,
                    scene_list_path=dcfg.DATASET.TESTS.LIST_PATH,
                    df=self.dcfg.DF,
                    padding=dcfg.DATASET.TESTS.PADDING,
                    min_overlap_score=dcfg.DATASET.TESTS.MIN_OVERLAP_SCORE,
                    max_overlap_score=dcfg.DATASET.TESTS.MAX_OVERLAP_SCORE,
                    max_resize=self.args.img_size,
                    augment_fn=build_augmentor(dcfg.DATASET.TESTS.AUGMENTATION_TYPE),
                    max_samples=dcfg.DATASET.TESTS.MAX_SAMPLES,
                    mode='test',
                    njobs=dcfg.NJOBS,
                    cfg=dcfg.DATASET.TESTS,
                )
                tests_datasets += datasets
            self.tests_datasets = ConcatDataset(tests_datasets)
            os.environ['TOTAL_TESTS_SAMPLES'] = str(len(self.tests_datasets))
            if self.gpuid == 0: print('TOTAL_TESTS_SAMPLES:', len(self.tests_datasets))

    def _setup_dataset(self, benchmark, data_root, npz_root, scene_list_path, df, padding,
                       min_overlap_score, max_overlap_score, max_resize, augment_fn,
                       max_samples, mode, njobs, cfg):

        with open(scene_list_path, 'r') as f:
            seq_names = [name.strip() for name in f.readlines()]

        with tqdm_joblib(tqdm(bar_format="{l_bar}{bar:3}{r_bar}", ncols=100,
                              desc=f'[GPU {self.gpuid}] load {mode} {benchmark:14} data',
                              total=len(seq_names), disable=int(self.gpuid) != 0)):
            datasets = Parallel(n_jobs=njobs)(
                delayed(lambda x: _build_dataset(
                    Benchmarks.get(benchmark),
                    root_dir=data_root,
                    npz_root=npz_root,
                    seq_name=x,
                    mode=mode,
                    min_overlap_score=min_overlap_score,
                    max_overlap_score=max_overlap_score,
                    max_resize=max_resize,
                    df=df,
                    padding=padding,
                    augment_fn=augment_fn,
                    max_samples=max_samples,
                    **cfg
                ))(seqname) for seqname in seq_names)
        return datasets

    def train_dataloader(self, *args, **kwargs):
        if 'EPOCH' not in os.environ: print(f'[GPU {self.gpuid}] Train Dataset Length: {len(self.train_datasets)}')
        sampler = BalanceSampler(data_source = self.train_datasets, gpuid = self.gpuid, gpus = self.gpus, trains = self.args.trains, maxlen = self.args.maxlen)
        return DataLoader(self.train_datasets, sampler=sampler, collate_fn=collate_fn, **self.train_loader_params)

    def valid_dataloader(self, *args, **kwargs):
        if 'EPOCH' not in os.environ: print(f'[GPU {self.gpuid}] Valid Dataset Length: {len(self.valid_datasets)}')
        sampler = ValidSampler(data_source = self.valid_datasets, gpuid = self.gpuid, gpus = self.gpus, maxlen = 1000)
        return DataLoader(self.valid_datasets, sampler=sampler, collate_fn=collate_fn, **self.valid_loader_params)

    def val_dataloader(self, *args, **kwargs):
        return self.valid_dataloader(*args, **kwargs)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.tests_datasets, collate_fn=collate_fn, **self.tests_loader_params)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _build_dataset(dataset: Dataset, *args, **kwargs):
    # noinspection PyCallingNonCallable
    return dataset(*args, **kwargs)
