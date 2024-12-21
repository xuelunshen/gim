# -*- coding: utf-8 -*-
# @Author  : xuelun
import os

import cv2
import csv
import math
import torch
import scipy.io
import warnings
import argparse
import numpy as np

from os import mkdir
from tqdm import tqdm
from copy import deepcopy
from os.path import join, exists
from torch.utils.data import DataLoader

from datasets.walk.video_streamer import VideoStreamer
from datasets.walk.video_loader import WALKDataset, collate_fn

from networks.mit_semseg.models import ModelBuilder, SegmentationModule

gray2tensor = lambda x: (torch.from_numpy(x).float() / 255)[None, None]
color2tensor = lambda x: (torch.from_numpy(x).float() / 255).permute(2, 0, 1)[None]

warnings.simplefilter("ignore", category=UserWarning)

methods = {'SIFT', 'GIM_GLUE', 'GIM_LOFTR', 'GIM_DKM'}

PALETTE = scipy.io.loadmat('weights/color150.mat')['colors']

CLS_DICT = {}  # {'person': 13, 'sky': 3}
with open('weights/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        name = row[5].split(";")[0]
        if name == 'screen':
            name = '_'.join(row[5].split(";")[:2])
        CLS_DICT[name] = int(row[0]) - 1

exclude = ['person', 'sky', 'car']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--gpu", type=int,
                        default=0, help='-1 for CPU')
    parser.add_argument("--range", type=int, nargs='+',
                        default=None,
                        help='Video Range for seconds')
    parser.add_argument('--scene_name', type=str,
                        default=None,
                        help='Scene (video) name')
    parser.add_argument('--method', type=str, choices=methods,
                        required=True,
                        help='Method name')
    parser.add_argument('--resize', action='store_true',
                        help='whether resize')
    parser.add_argument('--skip', type=int,
                        required=True,
                        help='Video skip frame: 1, 2, 3, ...')
    parser.add_argument('--watermarker', type=int, nargs='+',
                        default=None,
                        help='Watermarker Rectangle Range')
    opt = parser.parse_args()

    data_root = join('data', 'ZeroMatch')
    video_name = opt.scene_name.strip()
    video_path = join(data_root, 'video_1080p', video_name + '.mp4')

    # get real size of video
    vcap = cv2.VideoCapture(video_path)
    vwidth = vcap.get(3)  # float `width`
    vheight = vcap.get(4)  # float `height`
    fps = vcap.get(5)  # float `fps`
    end_range = math.floor(vcap.get(cv2.CAP_PROP_FRAME_COUNT) / fps - 300)
    vcap.release()

    fps = math.ceil(fps)
    opt.range = [300, end_range] if opt.range is None else opt.range
    opt.range = [0, -1] if video_name == 'Od-rKbC30TM' else opt.range  # for demo

    if fps <= 30:
        skip = [10, 20, 40][opt.skip]
    else:
        skip = [20, 40, 80][opt.skip]

    dump_dir = join(data_root, 'pseudo',
                    'WALK ' + opt.method +
                    ' [R] ' + '{}'.format('T' if opt.resize else 'F') +
                    ' [S] ' + '{:2}'.format(skip))
    if not exists(dump_dir): mkdir(dump_dir)
    debug_dir = join('dump', video_name + ' ' + opt.method)
    if opt.resize: debug_dir = debug_dir + ' Resize'
    if opt.debug and (not exists(debug_dir)): mkdir(debug_dir)

    # start process video
    gap = 10 if fps <= 30 else 20
    vs = VideoStreamer(basedir=video_path, resize=opt.resize, df=8, skip=gap, vrange=opt.range)

    # read the first frame
    rgb = vs[vs.listing[0]]
    width, height = rgb.shape[1], rgb.shape[0]

    # calculate ratio
    vratio = np.array([vwidth / width, vheight / height])[None]

    # set dump name
    scene_name =  f'{video_name} '
    scene_name += f'WH {width:4} {height:4} '
    scene_name += f'RG {vs.range[0]:4} {vs.range[1]:4} '
    scene_name += f'SP {skip} '
    scene_name += f'{len(video_name)}'

    save_dir = join(dump_dir, scene_name)

    device = torch.device('cuda:{}'.format(opt.gpu)) if opt.gpu >= 0 else torch.device('cpu')

    # initialize segmentation model
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='weights/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='weights/decoder_epoch_20.pth',
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).to(device).eval()
    old_segment_root = join(data_root, 'segment', opt.scene_name)
    new_segment_root = join(data_root, 'segment', opt.scene_name.strip())
    if not os.path.exists(new_segment_root):
        if os.path.exists(old_segment_root):
            os.rename(old_segment_root, new_segment_root)
        else:
            os.makedirs(new_segment_root, exist_ok=True)
    segment_root = new_segment_root

    model, detectAndCompute = None, None

    if opt.method == 'SIFT':
        model = cv2.SIFT_create(nfeatures=32400, contrastThreshold=1e-5)
        detectAndCompute = model.detectAndCompute

    elif opt.method == 'GIM_DKM':
        from networks.dkm.models.model_zoo.DKMv3 import DKMv3
        model = DKMv3(weights=None, h=672, w=896)
        checkpoints_path = join('weights', 'gim_dkm_100h.ckpt')
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)
        model = model.eval().to(device)

    elif opt.method == 'GIM_LOFTR':
        from networks.loftr.loftr import LoFTR
        from networks.loftr.misc import lower_config
        from networks.loftr.config import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.TEMP_BUG_FIX = True
        cfg.LOFTR.WEIGHT = 'weights/gim_loftr_50h.ckpt'
        cfg.LOFTR.FINE_CONCAT_COARSE_FEAT = False
        cfg = lower_config(cfg)
        model = LoFTR(cfg['loftr'])
        model = model.to(device)
        model = model.eval()

    elif opt.method == 'GIM_GLUE':
        from networks.lightglue.matching import Matching

        model = Matching()

        checkpoints_path = join('weights', 'gim_lightglue_100h.ckpt')
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        model.detector.load_state_dict(state_dict)

        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.model.load_state_dict(state_dict)

        model = model.to(device)
        model = model.eval()

    cache_dir = None
    if opt.resize:
        cache_dir = join(data_root, 'pseudo',
                         'WALK ' + 'GIM_DKM' +
                         ' [R] F' +
                         ' [S] ' + '{:2}'.format(skip),
                         scene_name)

    _w_ = width if opt.method == 'SIFT' or opt.method == 'GLUE' else 1600 # TODO: confirm DKM
    _h_ = height if opt.method == 'SIFT' or opt.method == 'GLUE' else 900 # TODO: confirm DKM

    ids = list(zip(vs.listing[:-skip // gap], vs.listing[skip // gap:]))

    # start matching and make pseudo labels
    nums = None
    idxs = None
    checkpoint = 0
    if not opt.debug:
        if exists(join(save_dir, 'nums.npy')) and exists(join(save_dir, 'idxs.npy')):
            with open(join(save_dir, 'nums.npy'), 'rb') as f:
                nums = np.load(f)
            with open(join(save_dir, 'idxs.npy'), 'rb') as f:
                idxs = np.load(f)
            assert len(nums) == len(idxs) == (len(os.listdir(save_dir)) - 2)
            whole = [str(x) + '.npy' for x in np.array(ids)]
            cache = [str(x) + '.npy' for x in idxs]
            leave = list(set(whole) - set(cache))
            if len(leave):
                leave = list(map(lambda x: int(x.rsplit('[')[-1].strip().split()[0]), leave))
                skip_id = np.array(sorted(leave))
                skip_id = (skip_id[1:] - skip_id[:-1]) // gap
                len_id = len(skip_id)
                if len_id == 0: exit(0)
                skip_id = [i for i in range(len_id) if skip_id[i:].sum() == (len_id - i)]
                if len(skip_id) == 0: exit(0)
                skip_id = skip_id[0]
                checkpoint = np.where(np.array(ids)[:, 0]==sorted(leave)[skip_id])[0][0]
                if len(nums) + skip_id > checkpoint: exit(0)
                assert checkpoint == len(nums) + skip_id
            else:
                exit(0)
        else:
            if not exists(save_dir): mkdir(save_dir)
            nums = np.array([])
            idxs = np.array([])
    datasets = WALKDataset(data_root, vs=vs, ids=ids, checkpoint=checkpoint, opt=opt)
    loader_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 5,
                     'pin_memory': True, 'drop_last': False}
    loader = DataLoader(datasets, collate_fn=collate_fn, **loader_params)
    for i, batch in enumerate(tqdm(loader, ncols=120, bar_format="{l_bar}{bar:3}{r_bar}",
                                   desc='{:11} - [{:5}, {:2}{}]'.format(video_name[:40], opt.method, skip, '*' if opt.resize else ''),
                                   total=len(loader), leave=False)):
        idx = batch['idx'].item()
        assert i == idx
        idx0 = batch['idx0'].item()
        idx1 = batch['idx1'].item()
        assert idx0 == ids[idx+checkpoint][0] and idx1 == ids[idx+checkpoint][1]

        # cache loaded image
        if not batch['rgb0_is_good'].item():
            img_path0 = batch['img_path0'][0]
            if not os.path.exists(img_path0):
                cv2.imwrite(img_path0, batch['rgb0'].squeeze(0).numpy())
        if not batch['rgb1_is_good'].item():
            img_path1 = batch['img_path1'][0]
            if not os.path.exists(img_path1):
                cv2.imwrite(img_path1, batch['rgb1'].squeeze(0).numpy())

        current_id = np.array([idx0, idx1])
        save_name = '{}.npy'.format(str(current_id))
        save_path = join(save_dir, save_name)
        if exists(save_path) and not opt.debug: continue

        rgb0 = batch['rgb0'].squeeze(0).numpy()
        rgb1 = batch['rgb1'].squeeze(0).numpy()
        _rgb0_, _rgb1_ = deepcopy(rgb0), deepcopy(rgb1)

        # get correspondeces in unresize image
        pt0, pt1 = None, None
        if opt.resize:
            cache_path = join(cache_dir, save_name)
            if not exists(cache_path): continue
            with open(cache_path, 'rb') as f:
                pts = np.load(f)
                pt0, pt1 = pts[:, :2], pts[:, 2:]

        # process first frame image
        xA0, xA1, yA0, yA1, hA, wA, wA_new, hA_new = None, None, None, None, None, None, None, None
        if opt.resize:
            # crop rgb0
            xA0 = math.floor(pt0[:, 0].min())
            xA1 = math.ceil(pt0[:, 0].max())
            yA0 = math.floor(pt0[:, 1].min())
            yA1 = math.ceil(pt0[:, 1].max())
            rgb0 = rgb0[yA0:yA1, xA0:xA1]
            hA, wA = rgb0.shape[:2]
            wA_new, hA_new = get_resized_wh(wA, hA, [_h_, _w_])
            wA_new, hA_new = get_divisible_wh(wA_new, hA_new, 8)
            rgb0 = cv2.resize(rgb0, (wA_new, hA_new), interpolation=cv2.INTER_AREA)

        # go on
        gray0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
        # semantic segmentation
        with torch.no_grad():
            seg_path0 = join(segment_root, '{}.npy'.format(idx0))
            if not os.path.exists(seg_path0):
                mask0 = segment(_rgb0_, device, segmentation_module)
                np.save(seg_path0, mask0)
            else:
                mask0 = np.load(seg_path0)

        # process next frame image
        xB0, xB1, yB0, yB1, hB, wB, wB_new, hB_new = None, None, None, None, None, None, None, None
        if opt.resize:
            # crop rgb1
            xB0 = math.floor(pt1[:, 0].min())
            xB1 = math.ceil(pt1[:, 0].max())
            yB0 = math.floor(pt1[:, 1].min())
            yB1 = math.ceil(pt1[:, 1].max())
            rgb1 = rgb1[yB0:yB1, xB0:xB1]
            hB, wB = rgb1.shape[:2]
            wB_new, hB_new = get_resized_wh(wB, hB, [_h_, _w_])
            wB_new, hB_new = get_divisible_wh(wB_new, hB_new, 8)
            rgb1 = cv2.resize(rgb1, (wB_new, hB_new), interpolation=cv2.INTER_AREA)

        # go on
        gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
        # semantic segmentation
        with torch.no_grad():
            seg_path1 = join(segment_root, '{}.npy'.format(idx1))
            if not os.path.exists(seg_path1):
                mask1 = segment(_rgb1_, device, segmentation_module)
                np.save(seg_path1, mask1)
            else:
                mask1 = np.load(seg_path1)

        if mask0.shape[:2] != _rgb0_.shape[:2]:
            mask0 = cv2.resize(mask0, _rgb0_.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        if mask1.shape != _rgb1_.shape[:2]:
            mask1 = cv2.resize(mask1, _rgb1_.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        if opt.resize:
            # resize mask0
            mask0 = mask0[yA0:yA1, xA0:xA1]
            mask0 = cv2.resize(mask0, (wA_new, hA_new), interpolation=cv2.INTER_NEAREST)
            # resize mask1
            mask1 = mask1[yB0:yB1, xB0:xB1]
            mask1 = cv2.resize(mask1, (wB_new, hB_new), interpolation=cv2.INTER_NEAREST)

        data = None
        if opt.method == 'SIFT':

            mask_0 = mask0 != CLS_DICT[exclude[0]]
            mask_1 = mask1 != CLS_DICT[exclude[0]]
            for cls in exclude[1:]:
                mask_0 = mask_0 & (mask0 != CLS_DICT[cls])
                mask_1 = mask_1 & (mask1 != CLS_DICT[cls])
            mask_0 = mask_0.astype(np.uint8)
            mask_1 = mask_1.astype(np.uint8)

            if mask_0.sum() == 0 or mask_1.sum() == 0: continue

            # keypoint detection and description
            kpts0, desc0 = detectAndCompute(rgb0, mask_0)
            if desc0 is None or desc0.shape[0] < 8: continue
            kpts0 = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts0])
            kpts0, desc0 = map(lambda x: torch.from_numpy(x).to(device).float(), [kpts0, desc0])
            desc0 = (desc0 / desc0.sum(dim=1, keepdim=True)).sqrt()

            # keypoint detection and description
            kpts1, desc1 = detectAndCompute(rgb1, mask_1)
            if desc1 is None or desc1.shape[0] < 8: continue
            kpts1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts1])
            kpts1, desc1 = map(lambda x: torch.from_numpy(x).to(device).float(), [kpts1, desc1])
            desc1 = (desc1 / desc1.sum(dim=1, keepdim=True)).sqrt()

            # mutual nearest matching and ratio filter
            matches = desc0 @ desc1.transpose(0, 1)
            mask = (matches == matches.max(dim=1, keepdim=True).values) & \
                   (matches == matches.max(dim=0, keepdim=True).values)
            # noinspection PyUnresolvedReferences
            valid, indices = mask.max(dim=1)
            ratio = torch.topk(matches, k=2, dim=1).values
            ratio = (-2 * ratio + 2).sqrt()
            # ratio = (ratio[:, 0] / ratio[:, 1]) < opt.mt
            ratio = (ratio[:, 0] / ratio[:, 1]) < 0.8
            valid = valid & ratio

            # get matched keypoints
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[indices[valid]]
            b_ids = torch.where(valid[None])[0]

            data = dict(
                m_bids = b_ids,
                mkpts0_f = mkpts0,
                mkpts1_f = mkpts1,
            )

        elif opt.method == 'GIM_DKM':

            mask_0 = mask0 != CLS_DICT[exclude[0]]
            mask_1 = mask1 != CLS_DICT[exclude[0]]
            for cls in exclude[1:]:
                mask_0 = mask_0 & (mask0 != CLS_DICT[cls])
                mask_1 = mask_1 & (mask1 != CLS_DICT[cls])
            mask_0 = mask_0.astype(np.uint8)
            mask_1 = mask_1.astype(np.uint8)

            if mask_0.sum() == 0 or mask_1.sum() == 0: continue

            img0 = rgb0 * mask_0[..., None]
            img1 = rgb1 * mask_1[..., None]

            width0, height0 = img0.shape[1], img0.shape[0]
            width1, height1 = img1.shape[1], img1.shape[0]

            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    img0 = torch.from_numpy(img0).permute(2, 0, 1).to(device)[None] / 255
                    img1 = torch.from_numpy(img1).permute(2, 0, 1).to(device)[None] / 255
                    dense_matches, dense_certainty = model.match(img0, img1)
                    sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 5000)
                mkpts0 = sparse_matches[:, :2]
                mkpts0 = torch.stack((width0 * (mkpts0[:, 0] + 1) / 2,
                                      height0 * (mkpts0[:, 1] + 1) / 2), dim=-1)
                mkpts1 = sparse_matches[:, 2:]
                mkpts1 = torch.stack((width1 * (mkpts1[:, 0] + 1) / 2,
                                      height1 * (mkpts1[:, 1] + 1) / 2), dim=-1)
                m_bids = torch.zeros(sparse_matches.shape[0], dtype=torch.long, device=device)

                data = dict(
                    m_bids = m_bids,
                    mkpts0_f = mkpts0,
                    mkpts1_f = mkpts1,
                )

        elif opt.method == 'GIM_LOFTR':

            mask_0 = mask0 != CLS_DICT[exclude[0]]
            mask_1 = mask1 != CLS_DICT[exclude[0]]
            for cls in exclude[1:]:
                mask_0 = mask_0 & (mask0 != CLS_DICT[cls])
                mask_1 = mask_1 & (mask1 != CLS_DICT[cls])
            mask_0 = mask_0.astype(np.uint8)
            mask_1 = mask_1.astype(np.uint8)

            if mask_0.sum() == 0 or mask_1.sum() == 0: continue

            mask_0 = cv2.resize(mask_0, None, fx=1/8, fy=1/8, interpolation=cv2.INTER_NEAREST)
            mask_1 = cv2.resize(mask_1, None, fx=1/8, fy=1/8, interpolation=cv2.INTER_NEAREST)

            data = dict(
                image0=gray2tensor(gray0),
                image1=gray2tensor(gray1),
                color0=color2tensor(rgb0),
                color1=color2tensor(rgb1),
                mask0=torch.from_numpy(mask_0)[None],
                mask1=torch.from_numpy(mask_1)[None],
            )

            with torch.no_grad():
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v
                        in data.items()}
                model(data)

        elif opt.method == 'GIM_GLUE':

            mask_0 = mask0 != CLS_DICT[exclude[0]]
            mask_1 = mask1 != CLS_DICT[exclude[0]]
            for cls in exclude[1:]:
                mask_0 = mask_0 & (mask0 != CLS_DICT[cls])
                mask_1 = mask_1 & (mask1 != CLS_DICT[cls])
            mask_0 = mask_0.astype(np.uint8)
            mask_1 = mask_1.astype(np.uint8)

            if mask_0.sum() == 0 or mask_1.sum() == 0: continue

            size0 = torch.tensor(gray0.shape[-2:][::-1])[None]
            size1 = torch.tensor(gray1.shape[-2:][::-1])[None]
            data = dict(
                gray0 = gray2tensor(gray0 * mask_0),
                gray1 = gray2tensor(gray1 * mask_1),
                size0 = size0,
                size1 = size1,
            )

            with torch.no_grad():
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v
                        in data.items()}
                pred = model(data)
                kpts0, kpts1 = pred['keypoints0'][0], pred['keypoints1'][0]
                matches = pred['matches'][0]
                if len(matches) == 0: continue

                mkpts0 = kpts0[matches[..., 0]]
                mkpts1 = kpts1[matches[..., 1]]
                m_bids = torch.zeros(matches[..., 0].size(), dtype=torch.long, device=device)

                data = dict(
                    m_bids = m_bids,
                    mkpts0_f = mkpts0,
                    mkpts1_f = mkpts1,
                )

        # auto remove watermarker
        kpts0 = data['mkpts0_f'].clone()  # (N, 2)
        kpts1 = data['mkpts1_f'].clone()  # (N, 2)
        moved = ~((kpts0 - kpts1).abs() < 1).min(dim=1).values  # (N)
        data['m_bids'] = data['m_bids'][moved]
        data['mkpts0_f'] = data['mkpts0_f'][moved]
        data['mkpts1_f'] = data['mkpts1_f'][moved]

        robust_fitting(data)
        if (data['inliers'] is None) or (sum(data['inliers'][0]) == 0): continue

        inliers = data['inliers'][0]

        if opt.debug:
            data.update(dict(
                # for debug visualization
                mask0 = mask0,
                mask1 = mask1,
                gray0 = gray0,
                gray1 = gray1,
                color0 = rgb0,
                color1 = rgb1,
                hw0_i = rgb0.shape[:2],
                hw1_i = rgb1.shape[:2],
                dataset_name = ['WALK'],
                scene_id = [video_name],
                pair_id = [[idx0, idx1]],
                imsize0=[[width, height]],
                imsize1=[[width, height]],
            ))
            out = fast_make_matching_robust_fitting_figure(data)
            cv2.imwrite(join(debug_dir, '{} {:8d} {:8d}.png'.format(scene_name, idx0, idx1)),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            continue

        if opt.resize:
            mkpts0_f = (data['mkpts0_f'].cpu().numpy()[inliers] * np.array([[wA/wA_new, hA/hA_new]]) + np.array([[xA0, yA0]])) * vratio
            mkpts1_f = (data['mkpts1_f'].cpu().numpy()[inliers] * np.array([[wB/wB_new, hB/hB_new]]) + np.array([[xB0, yB0]])) * vratio
        else:
            mkpts0_f = data['mkpts0_f'].cpu().numpy()[inliers] * vratio
            mkpts1_f = data['mkpts1_f'].cpu().numpy()[inliers] * vratio

        pts = np.concatenate([mkpts0_f, mkpts1_f], axis=1).astype(np.float32)
        nums = np.concatenate([nums, np.array([len(pts)])], axis=0) if len(nums) else np.array([len(pts)])
        idxs = np.concatenate([idxs, current_id[None]], axis=0) if len(idxs) else current_id[None]

        with open(save_path, 'wb') as f:
            np.save(f, pts)

        with open(join(save_dir, 'nums.npy'), 'wb') as f:
            np.save(f, nums)

        with open(join(save_dir, 'idxs.npy'), 'wb') as f:
            np.save(f, idxs)


def robust_fitting(data, b_id=0):
    m_bids = data['m_bids'].cpu().numpy()
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()

    mask = m_bids == b_id

    # noinspection PyBroadException
    try:
        _, mask = cv2.findFundamentalMat(kpts0[mask], kpts1[mask], cv2.USAC_MAGSAC, ransacReprojThreshold=0.5, confidence=0.999999, maxIters=100000)
        mask = (mask.ravel() > 0)[None]
    except:
        mask = None

    data.update(dict(inliers=mask))


def get_resized_wh(w, h, resize):
    nh, nw = resize
    sh, sw = nh / h, nw / w
    scale = min(sh, sw)
    w_new, h_new = int(round(w*scale)), int(round(h*scale))
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new = max((w // df), 1) * df
        h_new = max((h // df), 1) * df
    else:
        w_new, h_new = w, h
    return w_new, h_new


def read_deeplab_image(img, size=1920):
    width, height = img.shape[1], img.shape[0]

    if max(width, height) > size:
        if width > height:
            img = cv2.resize(img, (size, int(size * height / width)), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (int(size * width / height), size), interpolation=cv2.INTER_AREA)

    img = (torch.from_numpy(img).float() / 255).permute(2, 0, 1)[None]

    return img


def read_segmentation_image(img):
    img = read_deeplab_image(img, size=720)[0]
    img = img - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    img = img / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return img


def segment(rgb, device, segmentation_module):
    img_data = read_segmentation_image(rgb)
    singleton_batch = {'img_data': img_data[None].to(device)}
    output_size = img_data.shape[1:]
    # Run the segmentation at the highest resolution.
    scores = segmentation_module(singleton_batch, segSize=output_size)
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    return pred.cpu()[0].numpy().astype(np.uint8)


def getLabel(pair, idxs, nums, h5py_i, h5py_f):
    """
    Args:
        pair: [6965 6970]
        idxs: (N, 2)
        nums: (N,)
        h5py_i: (M, 2)
        h5py_f: (M, 2)

    Returns: pseudo_label (N, 4)
    """
    i, j = np.where(idxs == pair)
    if len(i) == 0: return None
    assert (len(i) == len(j) == 2) and (i[0] == i[1]) and (j[0] == 0) and (j[1] == 1)
    i = i[0]
    nums = nums[:i+1]
    idx0, idx1 = sum(nums[:-1]), sum(nums)

    mkpts0 = h5py_i[idx0:idx1]
    mkpts1 = h5py_f[idx0:idx1]  # (N, 2)

    return mkpts0, mkpts1


def fast_make_matching_robust_fitting_figure(data, b_id=0):
    b_mask = data['m_bids'] == b_id

    gray0 = data['gray0']
    gray1 = data['gray1']
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h, w = max(h0, h1), max(w0, w1)
    H, W = margin * 5 + h * 4, margin * 3 + w * 2

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w + margin, margin + w + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    color0 = data['color0']  # (rH, rW, 3)
    color1 = data['color1']  # (rH, rW, 3)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    img0 = np.stack([gray0] * 3, -1) * 0
    for cls in exclude: img0[data['mask0'] == CLS_DICT[cls]] = PALETTE[CLS_DICT[cls]]
    out[sh: sh + h0, wx[0]: wx[1]] = img0
    img1 = np.stack([gray1] * 3, -1) * 0
    for cls in exclude: img1[data['mask1'] == CLS_DICT[cls]] = PALETTE[CLS_DICT[cls]]
    out[sh: sh + h1, wx[2]: wx[3]] = img1

    # before outlier filtering
    sh = hx(row=3)
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        # display line end-points as circles
        c = (230, 216, 132)
        cv2.circle(out, (x0, y0+sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 3, c, -1, lineType=cv2.LINE_AA)

    # after outlier filtering
    if data['inliers'] is not None:
        sh = hx(row=4)
        inliers = data['inliers'][b_id]
        mkpts0, mkpts1 = np.round(kpts0).astype(int)[inliers], np.round(kpts1).astype(int)[inliers]
        out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
        out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            # display line end-points as circles
            c = (230, 216, 132)
            cv2.circle(out, (x0, y0+sh), 3, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + w, y1+sh), 3, c, -1, lineType=cv2.LINE_AA)

    # Big text.
    text = [
        f' ',
        f'#Matches {len(kpts0)}',
        f'#Matches {sum(data["inliers"][b_id]) if data["inliers"] is not None else 0}',
    ]
    sc = min(H / 640., 1.0)
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    fingerprint = [
        'Dataset: {}'.format(data['dataset_name'][b_id]),
        'Scene ID: {}'.format(data['scene_id'][b_id]),
        'Pair ID: {}'.format(data['pair_id'][b_id]),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id],
                                      data['imsize1'][b_id]),
    ]
    sc = min(H / 640., 1.0)
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    return out


if __name__ == '__main__':
    with torch.no_grad():
        main()
