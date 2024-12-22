# -*- coding: utf-8 -*-
# @Author  : xuelun

import torch
import random

import numpy as np
import kornia as K
import torch.nn.functional as F


from os.path import join

from datasets.dataset import RGBDDataset

from datasets.utils import split
from datasets.utils import read_images
from .utils import read_depth, read_pose

from datasets.walk.walk import degree_to_matrix, partial


class ScanNetDataset(RGBDDataset):
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

        # sample images by pairwise overlap ratio
        # load intrinsics and extrinsics

        self.root_dir = root_dir
        self.mode = mode
        self.scene_path = join(root_dir, seq_name)

        npz_path = join(npz_root, seq_name + '.npz')
        self.scene_id = seq_name
        self.scale = 1 / df

        with np.load(npz_path) as data:
            pair_ids = data['name']  # (n, 4)
            pair_scr = data['score']  # (n,)

            valid = (pair_ids[:,-2:] % 10).sum(axis=-1) == 0
            pair_ids = data['name'][valid]  # (n, 4)
            pair_scr = data['score'][valid]  # (n,)

            self.pair_ids, self.covision = [], []
            intervals = np.concatenate((np.arange(min_overlap_score, max_overlap_score, 0.1), [max_overlap_score]))
            sub_max_samples = split(max_samples, len(intervals) - 1)
            for low, upr, m in zip(intervals[:-1], intervals[1:], sub_max_samples):
                valid = (pair_scr > low) & (pair_scr <= upr)
                ids = pair_ids[valid]
                scr = pair_scr[valid]

                if len(ids) > m:
                    # sample max_samples
                    random_state = random.getstate()
                    np_random_state = np.random.get_state()
                    random.seed(3407)
                    np.random.seed(3407)
                    samples = random.sample(list(range(len(ids))), m)
                    random.setstate(random_state)
                    np.random.set_state(np_random_state)
                    # sample it
                    ids = ids[samples]
                    scr = scr[samples]

                if len(ids) > 0:
                    self.pair_ids += ids.tolist()
                    self.covision += scr.tolist()

        self.intrinsics = dict(np.load(kwargs.get('INTRINSIC_PATH'))) # (1513,) with (9,)

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train': assert max_resize is not None

        self.df = df
        self.max_resize = max_resize
        self.padding = padding

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        assert len(self.pair_ids) == len(self.covision)
        return len(self.pair_ids)

    def _read_abs_pose(self, scene_name, name):
        pth = join(self.root_dir, scene_name, 'pose', f'{name}.txt')
        return read_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)

        return np.matmul(pose1, np.linalg.inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        id_name = self.pair_ids[idx]
        covision = self.covision[idx]
        scene_name, scene_sub_name, img_name0, img_name1 = id_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        img_path0 = join(self.root_dir, scene_name, 'color', f'{img_name0}.jpg')
        img_path1 = join(self.root_dir, scene_name, 'color', f'{img_name1}.jpg')

        image0, color0, scale0, rands0, offset0, hlip0, vflip0, resize0, mask0 = read_images(
            img_path0, self.max_resize, self.df, self.padding,
            np.random.choice([self.augment_fn, None], p=[0.5, 0.5]),
            aug_prob=0.2 if self.mode == 'train' else 1.0, read_size=(640, 480))
        image1, color1, scale1, rands1, offset1, hlip1, vflip1, resize1, mask1 = read_images(
            img_path1, self.max_resize, self.df, self.padding,
            np.random.choice([self.augment_fn, None], p=[0.5, 0.5]),
            aug_prob=0.2 if self.mode == 'train' else 1.0, read_size=(640, 480))

        if self.mode == 'train' and random.uniform(0, 1) > 0.1:
            deg = 180
            angle = random.uniform(-deg, deg)

            h, w = image1.shape[-2:]
            Hq_aug = degree_to_matrix(-angle, (h - 1, w - 1))
            points_src = torch.tensor([[
                [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
            ]], dtype=torch.float32)
            # random sample four float numbers in [0, 0.3 * w] and [0, 0.3 * h]
            x_tl, x_bl, x_br, x_tr = random.sample(range(-int(w * 0.25), int(w * 0.25)), 4)
            y_tl, y_bl, y_br, y_tr = random.sample(range(-int(h * 0.25), int(h * 0.25)), 4)
            points_dst = torch.tensor([[
                [x_tl, y_tl], [w - x_tr, y_tr], [w - x_br, h - y_br], [x_bl, h - y_bl],
            ]], dtype=torch.float32)
            # compute perspective transform
            M_aug = K.geometry.get_perspective_transform(points_src, points_dst)[0]
            Hq_aug = torch.matmul(Hq_aug, M_aug)

            h, w = h * scale1[1], w * scale1[0]
            Hq_ori = degree_to_matrix(-angle, (h - 1, w - 1))
            points_src = torch.tensor([[
                [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
            ]], dtype=torch.float32)
            # random sample four float numbers in [0, 0.3 * w] and [0, 0.3 * h]
            x_tl, x_bl, x_br, x_tr = map(lambda x: x * scale1[0], [x_tl, x_bl, x_br, x_tr])
            y_tl, y_bl, y_br, y_tr = map(lambda y: y * scale1[1], [y_tl, y_bl, y_br, y_tr])
            points_dst = torch.tensor([[
                [x_tl, y_tl], [w - x_tr, y_tr], [w - x_br, h - y_br], [x_bl, h - y_bl],
            ]], dtype=torch.float32)
            # compute perspective transform
            M_ori = K.geometry.get_perspective_transform(points_src, points_dst)[0]
            Hq_ori = torch.matmul(Hq_ori, M_ori)

            # define perspective transform
            h, w = image1.shape[-2:]
            perspective = partial(K.geometry.warp_perspective, M=Hq_aug[None], dsize=(h, w), mode='bilinear')
            # apply homography on mask1
            mask1 = perspective(src=mask1[None, None].float())[0, 0].bool()
            # apply homography on color1
            color1 = perspective(src=color1[None])[0]
            # apply homography on image1
            image1 = perspective(src=image1[None])[0]
        else:
            Hq_aug = torch.eye(3, dtype=torch.float)
            Hq_ori = torch.eye(3, dtype=torch.float)

        depth0 = read_depth(join(self.root_dir, scene_name, 'depth', f'{img_name0}.png'), pad_to=(1600, 1600))
        depth1 = read_depth(join(self.root_dir, scene_name, 'depth', f'{img_name1}.png'), pad_to=(1600, 1600))

        K0 = K1 = torch.tensor(self.intrinsics[scene_name], dtype=torch.float).reshape(3, 3)

        # read image size
        imsize0 = torch.tensor([480, 640], dtype=torch.long)
        imsize1 = torch.tensor([480, 640], dtype=torch.long)
        resize0 = torch.tensor(resize0, dtype=torch.long)
        resize1 = torch.tensor(resize1, dtype=torch.long)

        # read and compute relative poses
        T_0to1 = self._compute_rel_pose(scene_name, img_name0, img_name1)
        T_0to1 = torch.tensor(T_0to1, dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        data = {
            # image 0
            'image0': image0, # (1, 3, h, w)
            'color0': color0,  # (1, h, w)
            'imsize0': imsize0, # (2) - 2:(h, w)
            'offset0': offset0, # [w_start, h_start] in resized scale
            'resize0': resize0, # (2) - 2:(h, w)
            'depth0': depth0,  # (h, w)
            'hflip0': hlip0,
            'vflip0': vflip0,

            # image 1
            'image1': image1,
            'color1': color1,
            'imsize1': imsize1,  # (2) - 2:[h, w]
            'offset1': offset1,  # [w_start, h_start] in resized scale
            'resize1': resize1, # (2) - 2:(h, w)
            'depth1': depth1,
            'hflip1': hlip1,
            'vflip1': vflip1,

            # image transform
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K0,  # (3, 3)
            'K1': K1,
            'Hq_aug': Hq_aug,  # (3, 3)
            'Hq_ori': Hq_ori,  # (3, 3)
            # pair information
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'rands0': rands0,
            'rands1': rands1,
            'dataset_name': 'ScanNet',
            'scene_id': self.scene_id,
            'pair_id': str(id_name),
            'pair_names': (str(img_name0),
                           str(img_name1)),
            'covisible0': covision,
            'covisible1': covision,
        }

        item = super(ScanNetDataset, self).__getitem__(idx)
        item.update(data)
        data = item

        if mask0 is not None:  # img_padding is True
            if self.scale:
                # noinspection PyArgumentList
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
            data.update({'mask0_i': mask0, 'mask1_i': mask1})

        return data
