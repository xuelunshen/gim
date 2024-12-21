# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from os import listdir
from pathlib import Path
from functools import reduce
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, isdir, exists

from datasets.dataset import RGBDDataset

from datasets.walk import cfg
from datasets.walk.utils import covision, intersected, read_images
from datasets.walk.utils import fast_make_matching_robust_fitting_figure

parse_mtd = lambda name: name.parent.stem.split()[1]
parse_skip = lambda name: int(str(name).split(os.sep)[-1].rpartition('SP')[-1].strip().rpartition(' ')[0])
parse_resize = lambda name: str(name).split(os.sep)[-2].rpartition('[R]')[-1].rpartition('[S]')[0].strip()

create_table = lambda x, y, w: dict(zip(np.round(x) + np.round(y) * w, list(range(len(x)))))


class WALKDataset(RGBDDataset):
    def __init__(self,
                 root_dir,          # data root dit
                 npz_root,          # data info, like, overlap, image_path, depth_path
                 seq_name,          # current sequence
                 mode,              # train or val or test
                 max_resize,        # max edge after resize
                 df,                # general is 8 for ResNet w/ pre 3-layers
                 padding,           # padding image for batch training
                 augment_fn,        # augmentation function
                 max_samples,       # max sample in current sequence
                 **kwargs):
        super().__init__()

        self.mode = mode
        self.root_dir = root_dir
        self.scene_path = join(root_dir, seq_name)

        pseudo_labels = kwargs.get('PSEUDO_LABELS', None)
        npz_paths = [join(npz_root, x) for x in pseudo_labels]
        npz_paths = [x for x in npz_paths if exists(x)]
        npz_names = [{d[:int(d.split()[-1])]: Path(path, d) for d in listdir(path) if isdir(join(path, d))} for path in npz_paths]
        npz_paths = [name_dict[seq_name] for name_dict in npz_names if seq_name in name_dict.keys()]

        self.propagating = kwargs.get('PROPAGATING', False)

        if self.propagating and len(npz_paths) != 24:
            print(f'{seq_name} has {len(npz_paths)} pseudo labels, but 24 are expected.')
            exit(0)

        self.scale = 1 / df
        self.scene_id = seq_name
        self.skips = sorted(list({parse_skip(name) for name in npz_paths}))
        self.resizes = sorted(list({parse_resize(name) for name in npz_paths}))
        self.methods = sorted(list({parse_mtd(name) for name in npz_paths}))[::-1]

        self.min_final_matches = kwargs.get('MIN_FINAL_MATCHES', None)
        self.min_filter_matches = kwargs.get('MIN_FILTER_MATCHES', None)

        pproot = kwargs.get('PROPAGATE_ROOT', None)
        ppid = ' '.join(self.methods + list(map(str, self.skips)) + self.resizes + [f'FM {self.min_filter_matches}', f'PM {self.min_final_matches}'])
        self.pproot = join(pproot, ppid, seq_name)

        if not self.propagating:
            assert exists(self.pproot)
        elif not exists(self.pproot):
            os.makedirs(self.pproot, exist_ok=True)

        image_root = kwargs.get('VIDEO_IMAGE_ROOT', None)
        self.image_root = join(image_root, seq_name)
        if not exists(self.image_root):
            os.makedirs(self.image_root, exist_ok=True)

        self.step = kwargs.get('STEP', None)
        self.pix_thr = kwargs.get('PIX_THR', None)
        self.fix_matches = kwargs.get('FIX_MATCHES', None)

        source_root = kwargs.get('SOURCE_ROOT', None)

        scap = cv2.VideoCapture(join(source_root, seq_name + '.mp4'))
        self.pseudo_size = [int(scap.get(3)), int(scap.get(4))]
        source_fps = int(scap.get(5))

        video_path = join(root_dir, seq_name + '.mp4')
        vcap = cv2.VideoCapture(video_path)
        self.frame_size = [int(vcap.get(3)), int(vcap.get(4))]

        if self.propagating:
            nums = {skip: [] for skip in self.skips}
            idxs = {skip: [] for skip in self.skips}
            self.path = {skip: [] for skip in self.skips}
            for npz_path in npz_paths:
                skip = parse_skip(npz_path)
                assert exists(npz_path / 'nums.npy')
                with open(npz_path / 'nums.npy', 'rb') as f:
                    npz = np.load(f)
                    nums[skip].append(npz)
                assert exists(npz_path / 'idxs.npy')
                with open(npz_path / 'idxs.npy', 'rb') as f:
                    npz = np.load(f)
                    idxs[skip].append(npz)
                self.path[skip].append(npz_path)

            ids1 = reduce(intersected, [idxs[nums > self.min_filter_matches] for nums, idxs in zip(nums[self.skips[-1]], idxs[self.skips[-1]])])
            continue1 = np.array([x in ids1[:, 0] for x in (ids1[:, 0] + self.skips[-1] * 1)])
            ids2 = reduce(intersected, idxs[self.skips[-2]])
            continue2 = np.array([x in ids2[:, 0] for x in ids1[:, 0]])
            continue2 = continue2 & np.array([x in ids2[:, 0] for x in (ids1[:, 0] + self.skips[-2] * 1)])
            ids3 = reduce(intersected, idxs[self.skips[-3]])
            continue3 = np.array([x in ids3[:, 0] for x in ids1[:, 0]])
            continue3 = continue3 & np.array([x in ids3[:, 0] for x in (ids1[:, 0] + self.skips[-3] * 1)])
            continue3 = continue3 & np.array([x in ids3[:, 0] for x in (ids1[:, 0] + self.skips[-3] * 2)])
            continue3 = continue3 & np.array([x in ids3[:, 0] for x in (ids1[:, 0] + self.skips[-3] * 3)])
            continues = continue1 & continue2 & continue3
            ids = ids1[continues]
            pair_ids = np.array(list(zip(ids[:, 0], np.clip(ids[:, 0]+self.step*self.skips[-1], a_min=ids[0, 0], a_max=ids[-1, 1])))) if self.step > 0 else ids
            pair_ids = pair_ids[(pair_ids[:, 1] - pair_ids[:, 0]) >= self.skips[-1]]
        else:
            pair_ids = np.array([tuple(map(int, x.split('.npy')[0].split('_'))) for x in os.listdir(self.pproot) if x.endswith('.npy')])

        if (max_samples > 0) and (len(pair_ids) > max_samples):
            random_state = random.getstate()
            np_random_state = np.random.get_state()
            random.seed(3407)
            np.random.seed(3407)
            pair_ids = pair_ids[sorted(np.random.randint(len(pair_ids), size=max_samples))]
            random.setstate(random_state)
            np.random.set_state(np_random_state)

        # remove unvalid pairs from self.pproot/bad_pairs.txt
        pair_ids = set(map(tuple, pair_ids.tolist()))

        if self.propagating:
            assert not exists(join(self.pproot, 'bad_pairs.txt'))

        if exists(join(self.pproot, 'bad_pairs.txt')):
            with open(join(self.pproot, 'bad_pairs.txt'), 'r') as f:
                unvalid_pairs = set([tuple(map(int, line.split())) for line in f.readlines()])
                self.unvalid_pairs_num = len(unvalid_pairs) if not self.propagating else 'N/A'
                pair_ids = pair_ids - unvalid_pairs

        self.valid_pairs_num = len(pair_ids) if not self.propagating else 'N/A'

        self.pair_ids = list(map(list, pair_ids))  # List[List[int, int]]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train': assert max_resize is not None

        self.df = df
        self.max_resize = max_resize
        self.padding = padding

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.pair_ids)

    def propagate(self, idx0, idx1, skips):
        """
        Args:
            idx0: (int) index of the first frame
            idx1: (int) index of the second frame
            skips: (List)

        Returns:
        """
        skip = skips[-1]  # 40
        indices = [skip * (i + 1) + idx0 for i in range((idx1 - idx0) // skip)]
        if (not indices) or (idx0 != indices[0]): indices = [idx0] + indices
        if idx1 != indices[-1]: indices = indices + [idx1]
        indices = list(zip(indices[:-1], indices[1:]))

        # [(N', 4), (N'', 4), ...]
        labels = []
        ids = [idx0]
        while indices:
            pair = indices.pop(0)  # (tuple)
            if pair[0] == pair[1]: break
            label = []
            if (pair[-1] - pair[0]) == skip:
                tmp = self.dump(skip, pair)
                if len(tmp) > 0: label.append(tmp)  # (ndarray) (N, 4)
            if skips[:-1]:
                _label_, id0, id1 = self.propagate(pair[0], pair[1], skips[:-1])
                if (id0, id1) == pair: label.append(_label_)  # (ndarray) (M, 4)
            if label:
                label = np.concatenate(label, axis=0)  # (ndarray) (N+M, 4)
                labels.append(label)
                ids += [pair[1]]
            if len(labels) > 1:
                _labels_ = self.link(labels[0], labels[1])
                if _labels_ is not None:
                    labels = [_labels_]
                    ids = [ids[0], ids[-1]]
                else:
                    labels.pop(-1)
                    ids.pop(-1)
                    indices = [(pair[0], pair[1]-skips[0])]

        if len(labels) == 1 and len(ids) == 2:
            return labels[0], ids[0], ids[-1]
        else:
            return None, None, None

    def link(self, label0, label1):
        """
        Args:
            label0: (ndarray) N x 4
            label1: (ndarray) M x 4

        Returns: (ndarray) (N', 4)
        """
        # get keypoints in left, middle and right frame
        left_t0 = label0[:, :2]  # (N, 2)
        mid_t0 = label0[:, 2:]  # (N, 2)
        mid_t1 = label1[:, :2]  # (M, 2)
        right_t1 = label1[:, 2:]  # (M, 2)

        mid0_table = create_table(mid_t0[:, 0], mid_t0[:, 1], self.pseudo_size[0])
        mid1_table = create_table(mid_t1[:, 0], mid_t1[:, 1], self.pseudo_size[0])

        keys = {*mid0_table} & {*mid1_table}

        i = np.array([mid0_table[k] for k in keys])
        j = np.array([mid1_table[k] for k in keys])

        # remove repeat matches
        ij = np.unique(np.vstack((i, j)), axis=1)

        if ij.shape[1] < self.min_final_matches: return None

        # get the new pseudo labels
        pseudo_label = np.concatenate([left_t0[ij[0]], right_t1[ij[1]]], axis=1)  # (N', 4)

        return pseudo_label

    def dump(self, skip, pair):
        """
        Args:
            skip:
            pair:

        Returns: pseudo_label (N, 4)
        """
        labels = []
        for path in self.path[skip]:
            p = path / '{}.npy'.format(str(np.array(pair)))
            if exists(p):
                with open(p, 'rb') as f:
                    labels.append(np.load(f))

        if len(labels) > 0: labels = np.concatenate(labels, axis=0).astype(np.float32)  # (N, 4)

        return labels

    def __getitem__(self, idx):
        idx0, idx1 = self.pair_ids[idx]

        pppath = join(self.pproot, '{}_{}.npy'.format(idx0, idx1))

        if self.propagating and exists(pppath):
            return None

        # check propagation
        if not self.propagating:
            assert exists(pppath), f'{pppath} does not exist'

        if not exists(pppath):
            pseudo_label, idx0, idx1 = self.propagate(idx0, idx1, self.skips)

            if idx1 - idx0 == self.skips[-1]:
                pseudo_label, idx0, idx1 = self.propagate(idx0, idx1, self.skips[:-1])

            if idx1 - idx0 == self.skips[-2]:
                pseudo_label, idx0, idx1 = self.propagate(idx0, idx1, self.skips[:-2])

            if pseudo_label is None:
                _idx0_, _idx1_ = self.pair_ids[idx]
                with open(join(self.pproot, 'bad_pairs.txt'), 'a') as f:
                    f.write('{} {}\n'.format(_idx0_, _idx1_))
                return None

            _, mask = cv2.findFundamentalMat(pseudo_label[:, :2], pseudo_label[:, 2:], cv2.USAC_MAGSAC, ransacReprojThreshold=1.0, confidence=0.999999, maxIters=1000)
            mask = mask.ravel() > 0
            pseudo_label = pseudo_label[mask]

            if len(pseudo_label) < 64 or (idx1 - idx0) == self.skips[-3]:
                _idx0_, _idx1_ = self.pair_ids[idx]
                with open(join(self.pproot, 'bad_pairs.txt'), 'a') as f:
                    f.write('{} {}\n'.format(_idx0_, _idx1_))
                return None
            else:
                with open(pppath, 'wb') as f:
                    np.save(f, np.concatenate((np.array([[idx0, idx1, idx0, idx1]]).astype(np.float32), pseudo_label), axis=0))
        else:
            with open(pppath, 'rb') as f:
                pseudo_label = np.load(f)
                idx0, idx1 = pseudo_label[0].astype(np.int64)[:2].tolist()
                pseudo_label = pseudo_label[1:]

        if self.propagating:
            return None

        pseudo_label *= (np.array(self.frame_size * 2) / np.array(self.pseudo_size * 2))[None]

        # get image
        img_path0 = join(self.image_root, '{}.png'.format(idx0))
        color0 = cv2.imread(img_path0)

        img_path1 = join(self.image_root, '{}.png'.format(idx1))
        color1 = cv2.imread(img_path1)

        width0, height0 = self.frame_size
        width1, height1 = self.frame_size

        left_upper_cornor = pseudo_label[:, :2].min(axis=0)
        left_low_corner = pseudo_label[:, :2].max(axis=0)
        left_corner = np.concatenate([left_upper_cornor, left_low_corner], axis=0)
        right_upper_cornor = pseudo_label[:, 2:].min(axis=0)
        right_low_corner = pseudo_label[:, 2:].max(axis=0)
        right_corner = np.concatenate([right_upper_cornor, right_low_corner], axis=0)

        # Prepare variables
        image0, color0, scale0, rands0, offset0, hlip0, vflip0, resize0, mask0 = read_images(
            None, self.max_resize, self.df, self.padding,
            np.random.choice([self.augment_fn, None], p=[0.5, 0.5]),
            aug_prob=1.0, is_left=True,
            upper_cornor=left_corner,
            read_size=self.frame_size, image=color0)
        image1, color1, scale1, rands1, offset1, hlip1, vflip1, resize1, mask1 = read_images(
            None, self.max_resize, self.df, self.padding,
            np.random.choice([self.augment_fn, None], p=[0.5, 0.5]),
            aug_prob=1.0, is_left=False,
            upper_cornor=right_corner,
            read_size=self.frame_size, image=color1)

        # warp keypoints by scale, offset and hlip
        pseudo_label = torch.tensor(pseudo_label, dtype=torch.float)
        left = (pseudo_label[:, :2] / scale0[None] - offset0[None])
        left[:, 0] = resize0[1] - 1 - left[:, 0] if hlip0 else left[:, 0]
        left[:, 1] = resize0[0] - 1 - left[:, 1] if vflip0 else left[:, 1]
        right = (pseudo_label[:, 2:] / scale1[None] - offset1[None])
        right[:, 0] = resize1[1] - 1 - right[:, 0] if hlip1 else right[:, 0]
        right[:, 1] = resize1[0] - 1 - right[:, 1] if vflip1 else right[:, 1]

        mask = (left[:, 0] >= 0) & (left[:, 0]*self.scale <= (resize0[1]*self.scale - 1)) & \
               (left[:, 1] >= 0) & (left[:, 1]*self.scale <= (resize0[0]*self.scale - 1)) & \
               (right[:, 0] >= 0) & (right[:, 0]*self.scale <= (resize1[1]*self.scale - 1)) & \
               (right[:, 1] >= 0) & (right[:, 1]*self.scale <= (resize1[0]*self.scale - 1))
        left, right = left[mask], right[mask]

        pseudo_label = torch.cat([left, right], dim=1)
        pseudo_label = torch.unique(pseudo_label, dim=0)

        fix_pseudo_label = torch.zeros(self.fix_matches, 4, dtype=pseudo_label.dtype)
        fix_pseudo_label[:len(pseudo_label)] = pseudo_label

        # read image size
        imsize0 = torch.tensor([height0, width0], dtype=torch.long)
        imsize1 = torch.tensor([height1, width1], dtype=torch.long)
        resize0 = torch.tensor(resize0, dtype=torch.long)
        resize1 = torch.tensor(resize1, dtype=torch.long)

        data = {
            # image 0
            'image0': image0,
            'color0': color0,
            'imsize0': imsize0,
            'offset0': offset0,
            'resize0': resize0,
            'depth0': torch.ones((1600, 1600), dtype=torch.float),
            'hflip0': hlip0,
            'vflip0': vflip0,

            # image 1
            'image1': image1,
            'color1': color1,
            'imsize1': imsize1,
            'offset1': offset1,
            'resize1': resize1,
            'depth1': torch.ones((1600, 1600), dtype=torch.float),
            'hflip1': hlip1,
            'vflip1': vflip1,

            # image transform
            'pseudo_labels': fix_pseudo_label,
            'gt': False,
            'zs': True,

            # image transform
            'T_0to1': torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float),
            'T_1to0': torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float),
            'K0': torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float),
            'K1': torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float),
            # pair information
            'scale0': scale0 / scale0,
            'scale1': scale1 / scale1,
            'rands0': rands0,
            'rands1': rands1,
            'dataset_name': 'WALK',
            'scene_id': '{:30}'.format(self.scene_id[:min(30, len(self.scene_id)-1)]),
            'pair_id': f'{idx0}-{idx1}',
            'pair_names': ('{}.png'.format(idx0),
                           '{}.png'.format(idx1)),
            'covisible0': covision(pseudo_label[:, :2], resize0).item(),
            'covisible1': covision(pseudo_label[:, 2:], resize1).item(),
        }

        item = super(WALKDataset, self).__getitem__(idx)
        item.update(data)
        data = item

        if mask0 is not None:
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
    parser = ArgumentParser()
    parser.add_argument('seq_names', type=str, nargs='+')
    args = parser.parse_args()

    train_cfg = cfg.DATASET.TRAIN

    base_input = {
        'df': 8,
        'mode': 'train',
        'augment_fn': None,
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

    if os.path.isfile(args.seq_names[0]):
        with open(args.seq_names[0], 'r') as f:
            seq_names = [line.strip() for line in f.readlines()]
    else:
        seq_names = args.seq_names

    for seq_name in seq_names:
        input_ = {
            **base_input,
            **cfg_input,
            'root_dir': cfg_input['DATA_ROOT'],
            'npz_root': cfg_input['NPZ_ROOT'],
            'seq_name': seq_name
        }
        
        dataset = WALKDataset(**input_)

        random.seed(3407)
        np.random.seed(3407)

        samples = list(range(len(dataset)))
        num = 10
        samples = random.sample(samples, num)
        for idx_ in tqdm(samples[:num], ncols=80, bar_format="{l_bar}{bar:3}{r_bar}", total=num,
                         desc=f'[ {seq_name[:min(10, len(seq_name)-1)]:<10} ] [ {dataset.valid_pairs_num:<5} / {dataset.valid_pairs_num+dataset.unvalid_pairs_num:<5} ]',):
            data_ = dataset[idx_]

            if data_ is None: continue

            pseudo_labels_ = data_['pseudo_labels']
            mask_ = pseudo_labels_.sum(dim=1) > 0
            pseudo_label_ = pseudo_labels_[mask_].cpu().numpy()
            data_['mkpts0_f'] = pseudo_label_[:, :2]
            data_['mkpts1_f'] = pseudo_label_[:, 2:]
            data_['hw0_i'] = data_['image0'].shape[-2:]
            data_['hw1_i'] = data_['image1'].shape[-2:]
            data_['image0'] = data_['image0'][None]
            data_['image1'] = data_['image1'][None]
            data_['color0'] = data_['color0'][None]
            data_['color1'] = data_['color1'][None]
            idx0_, idx1_ = data_['pair_id'].split('-')
            idx0_, idx1_ = map(int, [idx0_, idx1_])

            out = fast_make_matching_robust_fitting_figure(data_, transpose=True)
            save_dir = Path('dump/walk') / seq_name
            if not exists(save_dir): save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(join(save_dir, '{:8d} [{}] {:8d} {:3d}.png'.format(
                idx0_,
                datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S %f')[:-3],
                idx1_,
                idx1_ - idx0_
            )), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
