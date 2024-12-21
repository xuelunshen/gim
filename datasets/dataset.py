# -*- coding: utf-8 -*-
# @Author  : xuelun

import torch

from torch.utils.data import Dataset


class RGBDDataset(Dataset):
    def __getitem__(self, idx):

        data = {
            # image 0
            'image0': None,
            'color0': None,
            'imsize0': None,
            'resize0': None,

            # image 1
            'image1': None,
            'color1': None,
            'imsize1': None,
            'resize1': None,

            'pseudo_labels': torch.zeros((100000, 4), dtype=torch.float),
            'gt': True,
            'zs': False,

            # image transform
            'T_0to1': None,
            'T_1to0': None,
            'K0': None,
            'K1': None,
            # pair information
            'scale0': None,
            'scale1': None,
            'dataset_name': None,
            'scene_id': None,
            'pair_id': None,
            'pair_names': None,
            'covisible0': None,
            'covisible1': None,
            # ETH3D dataset
            'K0_': torch.zeros(12, dtype=torch.float),
            'K1_': torch.zeros(12, dtype=torch.float),
            # Hq
            'Hq_aug': torch.eye(3, dtype=torch.float),
            'Hq_ori': torch.eye(3, dtype=torch.float),
        }

        return data
