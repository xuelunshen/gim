# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import cv2
import torch

from os.path import join
from torch.utils.data import Dataset


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class WALKDataset(Dataset):

    def __init__(self, data_root, vs, ids, checkpoint, opt):
        super().__init__()

        self.vs = vs
        self.ids = ids[checkpoint:]

        old_image_root = join(data_root, 'image_1080p', opt.scene_name)
        new_image_root = join(data_root, 'image_1080p', opt.scene_name.strip())
        if not os.path.exists(new_image_root):
            if os.path.exists(old_image_root):
                os.rename(old_image_root, new_image_root)
            else:
                os.makedirs(new_image_root, exist_ok=True)
        self.image_root = new_image_root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx0, idx1 = self.ids[idx]

        # get image
        img_path0 = join(self.image_root, '{}.png'.format(idx0))
        if not os.path.exists(img_path0):
            rgb0 = self.vs[idx0]
            rgb0_is_good = False
        else:
            rgb0 = cv2.imread(img_path0)
            rgb0_is_good = True
            if rgb0 is None:
                rgb0 = self.vs[idx0]
                rgb0_is_good = False

        img_path1 = join(self.image_root, '{}.png'.format(idx1))
        if not os.path.exists(img_path1):
            rgb1 = self.vs[idx1]
            rgb1_is_good = False
        else:
            rgb1 = cv2.imread(img_path1)
            rgb1_is_good = True
            if rgb1 is None:
                rgb1 = self.vs[idx1]
                rgb1_is_good = False

        return {'idx': idx, 'idx0': idx0, 'idx1': idx1, 'rgb0': rgb0, 'rgb1': rgb1,
                'img_path0': img_path0, 'img_path1': img_path1,
                'rgb0_is_good':rgb0_is_good, 'rgb1_is_good': rgb1_is_good}
