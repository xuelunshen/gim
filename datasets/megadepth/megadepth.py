# -*- coding: utf-8 -*-
# @Author  : xuelun

import torch
import random
import imagesize
import numpy as np
import kornia as K
import torch.nn.functional as F


from os.path import join

from datasets.dataset import RGBDDataset

from datasets.utils import split
from datasets.utils import read_images
from datasets.megadepth.utils import read_megadepth_depth as read_depth

from datasets.walk.walk import degree_to_matrix, partial


class MegaDepthDataset(RGBDDataset):
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

        npz = np.load(npz_path, allow_pickle=True)
        self.mat = npz['overlap_matrix']

        self.pair_infos = []
        intervals = np.concatenate((np.arange(min_overlap_score, max_overlap_score, 0.1), [max_overlap_score]))
        sub_max_samples = split(max_samples, len(intervals) - 1)
        for low, upr, m in zip(intervals[:-1], intervals[1:], sub_max_samples):
            valid = np.logical_and(self.mat > low, self.mat <= upr)
            pair_infos = np.unique(np.sort(np.vstack(np.where(valid)).transpose(), axis=1), axis=0)
            covision0 = self.mat[tuple(pair_infos.transpose()[[0, 1]])]
            covision1 = self.mat[tuple(pair_infos.transpose()[[1, 0]])]
            valid = (covision0 > low) & (covision0 <= upr) &\
                    (covision1 > low) & (covision1 <= upr)
            pair_infos = pair_infos[valid]
            # above code is equivalent to code below
            # pair1 = np.vstack(np.where(valid)).transpose() # [n, 2]
            # pair2 = np.sort(pair1, axis=1)
            # self.pair_infos = np.unique(pair2, axis=0)
            if len(pair_infos) > m:
                random_state = random.getstate()
                np_random_state = np.random.get_state()
                random.seed(3407)
                np.random.seed(3407)
                pair_infos=random.sample(pair_infos.tolist(), m)
                random.setstate(random_state)
                np.random.set_state(np_random_state)
            else:
                pair_infos = pair_infos.tolist()

            if len(pair_infos) > 0: self.pair_infos += pair_infos

        self.image_paths = npz['image_paths']  # 图像路径 - [N]
        self.depth_paths = npz['depth_paths']  # 深度图 - [N]
        self.intrinsics = npz['intrinsics']  # 相机内参 - [N]
        self.poses = npz['poses']  # 相机外参 - [N]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train': assert max_resize is not None

        self.df = df
        self.max_resize = max_resize
        self.padding = padding

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        idx0, idx1 = self.pair_infos[idx]

        covisible0 = self.mat[idx0, idx1]
        covisible1 = self.mat[idx1, idx0]

        img_name0 = self.image_paths[idx0]
        img_name1 = self.image_paths[idx1]

        img_path0 = join(self.root_dir, img_name0)
        img_path1 = join(self.root_dir, img_name1)

        width0, height0 = imagesize.get(img_path0)
        width1, height1 = imagesize.get(img_path1)

        image0, color0, scale0, rands0, offset0, hlip0, vflip0, resize0, mask0 = read_images(
            img_path0, self.max_resize, self.df, self.padding,
            np.random.choice([self.augment_fn, None], p=[0.5, 0.5]),
            aug_prob=0.5 if self.mode == 'train' else 1.0)
        image1, color1, scale1, rands1, offset1, hlip1, vflip1, resize1, mask1 = read_images(
            img_path1, self.max_resize, self.df, self.padding,
            np.random.choice([self.augment_fn, None], p=[0.5, 0.5]),
            aug_prob=0.5 if self.mode == 'train' else 1.0)

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

        depth0 = read_depth(join(self.root_dir, self.depth_paths[idx0]), pad_to=(1600, 1600))
        depth1 = read_depth(join(self.root_dir, self.depth_paths[idx1]), pad_to=(1600, 1600))

        depth0 = torch.tensor(depth0, dtype=torch.float)
        depth1 = torch.tensor(depth1, dtype=torch.float)

        K0 = torch.tensor(self.intrinsics[idx0].copy(), dtype=torch.float).reshape(3, 3)
        K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)

        # read image size
        imsize0 = torch.tensor([height0, width0], dtype=torch.long)
        imsize1 = torch.tensor([height1, width1], dtype=torch.long)
        resize0 = torch.tensor(resize0, dtype=torch.long)
        resize1 = torch.tensor(resize1, dtype=torch.long)

        # read and compute relative poses
        WtoC0 = torch.tensor(self.poses[idx0],dtype=torch.float)
        WtoC1 = torch.tensor(self.poses[idx1],dtype=torch.float)
        T_0to1 = WtoC1 @ WtoC0.inverse()
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
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': f'{idx0}-{idx1}',
            'pair_names': (img_name0,
                           img_name1),
            'covisible0': covisible0,
            'covisible1': covisible1,
        }

        item = super(MegaDepthDataset, self).__getitem__(idx)
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


if __name__ == '__main__':
    import os
    import cv2
    from tqdm import tqdm
    from pathlib import Path
    from datetime import datetime
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--seq_name', type=str, required=True,)
    args = parser.parse_args()

    root_dir = 'data/MegaDepth/'
    npz_root = join(root_dir, 'output')

    seq_name = args.seq_name

    mode = 'train'
    min_overlap_score = 0.1
    max_overlap_score = 0.7
    max_resize = [840, 840]
    df = 8
    padding = True
    augment_fn = None
    max_samples = 21

    input = dict(
        root_dir = root_dir,
        npz_root = npz_root,
        seq_name = seq_name,
        mode = mode,
        min_overlap_score = min_overlap_score,
        max_overlap_score = max_overlap_score,
        max_resize = max_resize,
        df = df,
        padding = padding,
        augment_fn = augment_fn,
        max_samples = max_samples,
    )

    dataset = MegaDepthDataset(**input)

    random.seed(3407)
    np.random.seed(3407)

    samples = list(range(len(dataset)))
    num = min(len(dataset), 10e9)
    samples = random.sample(samples, num)
    for idx in tqdm(samples[:num], ncols=80, bar_format="{l_bar}{bar:3}{r_bar}", total=num,
                    desc=f'[ {seq_name[:min(10, len(seq_name) - 1)]:<10} ] [ {len(dataset):<5} ]', ):
        data = dataset[idx]

        idx0, idx1 = data['pair_id'].split('-')
        idx0, idx1 = map(int, [idx0, idx1])

        color0 = (data['color0'].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
        color1 = (data['color1'].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)

        save_dir = Path('../visualization') / seq_name
        if not os.path.exists(save_dir): save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(join(save_dir, '{:8d} [{}] {:8d}.png'.format(
            idx0,
            datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S %f')[:-3],
            idx1,
        )), cv2.cvtColor(np.concatenate((color0, color1), axis=1), cv2.COLOR_RGB2BGR))
