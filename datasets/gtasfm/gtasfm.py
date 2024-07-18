# -*- coding: utf-8 -*-
# @Author  : xuelun

import glob
import torch
import imagesize
import torch.nn.functional as F


from os.path import join

from torch.utils.data import Dataset

from datasets.utils import read_images


class GTASfMDataset(Dataset):
    def __init__(self,
                 root_dir,          # data root dit
                 npz_root,          # data info, like, overlap, image_path, depth_path
                 seq_name,          # current sequence
                 mode,              # train or val or test
                 min_overlap_score,
                 max_overlap_score,
                 max_resize,        # max edge after resize
                 df,                # general is 8 for ResNet w/ pre 3-layers
                 padding,           # padding image for batch training
                 augment_fn,        # augmentation function
                 max_samples,       # max sample in current sequence
                 **kwargs):
        super().__init__()

        self.root = join('zeb', seq_name)

        paths = glob.glob(join(self.root, '*.txt'))

        lines = []
        for path in paths:
            with open(path, 'r') as file:
                scene_id = path.rpartition('/')[-1].rpartition('.')[0].split('-')[0]
                line = file.readline().strip().split()
                lines.append([scene_id] + line)

        self.pairs = sorted(lines)

        self.scale = 1 / df

        self.df = df
        self.max_resize = max_resize
        self.padding = padding

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        scene_id = pair[0]

        img_name0 = pair[1]
        img_name1 = pair[2]

        img_path0 = join(self.root, '{}-{}.png'.format(scene_id, img_name0))
        img_path1 = join(self.root, '{}-{}.png'.format(scene_id, img_name1))

        width0, height0 = imagesize.get(img_path0)
        width1, height1 = imagesize.get(img_path1)

        image0, color0, scale0, resize0, mask0 = read_images(
            img_path0, self.max_resize, self.df, self.padding, None)
        image1, color1, scale1, resize1, mask1 = read_images(
            img_path1, self.max_resize, self.df, self.padding, None)

        K0 = torch.tensor(list(map(float, pair[5:14])), dtype=torch.float).reshape(3, 3)
        K1 = torch.tensor(list(map(float, pair[14:23])), dtype=torch.float).reshape(3, 3)

        # read image size
        imsize0 = torch.tensor([height0, width0], dtype=torch.long)
        imsize1 = torch.tensor([height1, width1], dtype=torch.long)
        resize0 = torch.tensor(resize0, dtype=torch.long)
        resize1 = torch.tensor(resize1, dtype=torch.long)

        # read and compute relative poses
        T_0to1 = torch.tensor(list(map(float, pair[23:])), dtype=torch.float).reshape(4, 4)

        data = {
            # image 0
            'image0': image0, # (1, 3, h, w)
            'color0': color0,  # (1, h, w)
            'imsize0': imsize0, # (2) - 2:(h, w)
            'resize0': resize0, # (2) - 2:(h, w)

            # image 1
            'image1': image1,
            'color1': color1,
            'imsize1': imsize1,  # (2) - 2:[h, w]
            'resize1': resize1, # (2) - 2:(h, w)

            # image transform
            'T_0to1': T_0to1,  # (4, 4)
            'K0': K0,  # (3, 3)
            'K1': K1,
            # pair information
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'GTA-SfM',
            'scene_id': scene_id,
            'pair_id': f'{idx}-{idx}',
            'pair_names': (img_name0,
                           img_name1),
            'covisible0': float(pair[3]),
            'covisible1': float(pair[4]),
        }

        if mask0 is not None:  # img_padding is True
            if self.scale:
                # noinspection PyArgumentList
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            # noinspection PyUnboundLocalVariable
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
