# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from datasets.walk import cfg
from datasets.walk.walk import WALKDataset


def propagate(loader, seq):
    for i, _ in enumerate(tqdm(
            loader, ncols=80, bar_format="{l_bar}{bar:3}{r_bar}", total=len(loader),
            desc=f'[ {seq[:min(10, len(seq)-1)]:<10} ] [ {len(loader):<5} ]')):
        continue


def init_dataset(seq_name_):
    train_cfg = cfg.DATASET.TRAIN
    
    base_input = {
        'df': 8,
        'mode': 'train',
        'augment_fn': None,
        'PROPAGATING': True,
        'seq_name': seq_name_,
        'max_resize': [1280, 720],
        'padding': cfg.DATASET.TRAIN.PADDING,
        'max_samples': cfg.DATASET.TRAIN.MAX_SAMPLES,
        'min_overlap_score': cfg.DATASET.TRAIN.MIN_OVERLAP_SCORE,
        'max_overlap_score': cfg.DATASET.TRAIN.MAX_OVERLAP_SCORE
    }
    
    cfg_input = {
        k: getattr(train_cfg, k) 
        for k in [
            'DATA_ROOT', 'NPZ_ROOT', 'STEP', 'PIX_THR', 'FIX_MATCHES', 'SOURCE_ROOT',
            'MAX_CANDIDATE_MATCHES', 'MIN_FINAL_MATCHES', 'MIN_FILTER_MATCHES',
            'VIDEO_IMAGE_ROOT', 'PROPAGATE_ROOT', 'PSEUDO_LABELS'
        ]
    }
    
    # 合并配置
    input_ = {
        **base_input,
        **cfg_input,
        'root_dir': cfg_input['DATA_ROOT'],
        'npz_root': cfg_input['NPZ_ROOT']
    }
    
    dataset = WALKDataset(**input_)

    return dataset


# noinspection PyUnusedLocal
def collate_fn(batch):
    return None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('seq_names', type=str, nargs='+')
    args = parser.parse_args()

    if os.path.isfile(args.seq_names[0]):
        with open(args.seq_names[0], 'r') as f:
            seq_names = [line.strip() for line in f.readlines()]
    else:
        seq_names = args.seq_names

    for seq_name in seq_names:

        dataset_ = init_dataset(seq_name)

        loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 3,
                         'pin_memory': True, 'drop_last': False}
        loader_ = DataLoader(dataset_, collate_fn=collate_fn, **loader_params)

        propagate(loader_, seq_name)
