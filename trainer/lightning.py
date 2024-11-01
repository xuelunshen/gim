# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from collections import OrderedDict

from tools.comm import all_gather
from tools.misc import lower_config, flattenList
from tools.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors


class Trainer(pl.LightningModule):

    def __init__(self, pcfg, tcfg, dcfg, ncfg):
        super().__init__()

        self.save_hyperparameters()
        self.pcfg = pcfg
        self.tcfg = tcfg
        self.ncfg = ncfg
        ncfg = lower_config(ncfg)

        detector = model = None
        if pcfg.weight == 'gim_dkm':
            from networks.dkm.models.model_zoo.DKMv3 import DKMv3
            detector = None
            model = DKMv3(None, 540, 720, upsample_preds=True)
            model.h_resized = 660
            model.w_resized = 880
            model.upsample_preds = True
            model.upsample_res = (1152, 1536)
            model.use_soft_mutual_nearest_neighbours = False
        elif pcfg.weight == 'gim_loftr':
            from networks.loftr.loftr import LoFTR as MODEL
            detector = None
            model = MODEL(ncfg['loftr'])
        elif pcfg.weight == 'gim_lightglue':
            from networks.lightglue.superpoint import SuperPoint
            from networks.lightglue.models.matchers.lightglue import LightGlue
            detector = SuperPoint({
                'max_num_keypoints': 2048,
                'force_num_keypoints': True,
                'detection_threshold': 0.0,
                'nms_radius': 3,
                'trainable': False,
            })
            model = LightGlue({
                'filter_threshold': 0.1,
                'flash': False,
                'checkpointed': True,
            })
        elif pcfg.weight == 'root_sift':
            detector = None
            model = None

        self.detector = detector
        self.model = model

        checkpoints_path = ncfg['loftr']['weight']
        if ncfg['loftr']['weight'] is not None:
            state_dict = torch.load(checkpoints_path, map_location='cpu')
            if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']

            if pcfg.weight == 'gim_dkm':
                for k in list(state_dict.keys()):
                    if k.startswith('model.'):
                        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
                    if 'encoder.net.fc' in k:
                        state_dict.pop(k)
            elif pcfg.weight == 'gim_lightglue':
                for k in list(state_dict.keys()):
                    if k.startswith('model.'):
                        state_dict.pop(k)
                    if k.startswith('superpoint.'):
                        state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
                self.detector.load_state_dict(state_dict)
                state_dict = torch.load(checkpoints_path, map_location='cpu')
                if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('superpoint.'):
                        state_dict.pop(k)
                    if k.startswith('model.'):
                        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)

            self.model.load_state_dict(state_dict)
            print('Load weights {} success'.format(ncfg['loftr']['weight']))

    def compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.tcfg)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(batch['scene_id'], *batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'covisible0': batch['covisible0'],
            'covisible1': batch['covisible1'],
            'Rot': batch['Rot'],
            'Tns': batch['Tns'],
            'Rot1': batch['Rot1'],
            'Tns1': batch['Tns1'],
            't_errs2': batch['t_errs2'],
        }
        return metrics

    def inference(self, data):
        if self.pcfg.weight == 'gim_dkm':
            self.gim_dkm_inference(data)
        elif self.pcfg.weight == 'gim_loftr':
            self.gim_loftr_inference(data)
        elif self.pcfg.weight == 'gim_lightglue':
            self.gim_lightglue_inference(data)
        elif self.pcfg.weight == 'root_sift':
            self.root_sift_inference(data)

    def gim_dkm_inference(self, data):
        dense_matches, dense_certainty = self.model.match(data['color0'], data['color1'])
        sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)
        hw0_i = data['color0'].shape[2:]
        hw1_i = data['color1'].shape[2:]
        height0, width0 = data['imsize0'][0]
        height1, width1 = data['imsize1'][0]
        kpts0 = sparse_matches[:, :2]
        kpts0 = torch.stack((width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
        kpts1 = sparse_matches[:, 2:]
        kpts1 = torch.stack((width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)

        b_ids = torch.where(mconf[None])[0]
        mask = mconf > 0

        data.update({
            'hw0_i': hw0_i,
            'hw1_i': hw1_i,
            'mkpts0_f': kpts0[mask],
            'mkpts1_f': kpts1[mask],
            'm_bids': b_ids,
            'mconf': mconf[mask],
        })

    def gim_loftr_inference(self, data):
        self.model(data)

    def gim_lightglue_inference(self, data):
        hw0_i = data['color0'].shape[2:]
        hw1_i = data['color1'].shape[2:]

        pred = {}
        pred.update({k+'0': v for k, v in self.detector({
            "image": data["image0"],
            "image_size": data["resize0"][:, [1, 0]],
        }).items()})
        pred.update({k+'1': v for k, v in self.detector({
            "image": data["image1"],
            "image_size": data["resize1"][:, [1, 0]],
        }).items()})
        pred.update(self.model({**pred, **data}))

        bs = data['image0'].size(0)
        mkpts0_f = torch.cat([kp * s for kp, s in zip(pred['keypoints0'], data['scale0'][:, None])])
        mkpts1_f = torch.cat([kp * s for kp, s in zip(pred['keypoints1'], data['scale1'][:, None])])
        m_bids = torch.nonzero(pred['keypoints0'].sum(dim=2) > -1)[:, 0]
        matches = pred['matches']
        mkpts0_f = torch.cat([mkpts0_f[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        mkpts1_f = torch.cat([mkpts1_f[m_bids == b_id][matches[b_id][..., 1]] for b_id in range(bs)])
        m_bids = torch.cat([m_bids[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        mconf = torch.cat(pred['scores'])

        data.update({
            'hw0_i': hw0_i,
            'hw1_i': hw1_i,
            'mkpts0_f': mkpts0_f,
            'mkpts1_f': mkpts1_f,
            'm_bids': m_bids,
            'mconf': mconf,
        })

    def root_sift_inference(self, data):
        # matching two images by sift
        image0 = data['color0'].squeeze().permute(1, 2, 0).cpu().numpy() * 255
        image1 = data['color1'].squeeze().permute(1, 2, 0).cpu().numpy() * 255

        image0 = cv2.cvtColor(image0.astype(np.uint8), cv2.COLOR_RGB2BGR)
        image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_RGB2BGR)

        H0, W0 = image0.shape[:2]
        H1, W1 = image1.shape[:2]

        sift0 = cv2.SIFT_create(nfeatures=H0*W0//64, contrastThreshold=1e-5)
        sift1 = cv2.SIFT_create(nfeatures=H1*W1//64, contrastThreshold=1e-5)

        kpts0, desc0 = sift0.detectAndCompute(image0, None)
        kpts1, desc1 = sift1.detectAndCompute(image1, None)
        kpts0 = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts0])
        kpts1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts1])

        kpts0, desc0, kpts1, desc1 = map(lambda x: torch.from_numpy(x).cuda().float(), [kpts0, desc0, kpts1, desc1])
        desc0, desc1 = map(lambda x: (x / x.sum(dim=1, keepdim=True)).sqrt(), [desc0, desc1])

        matches = desc0 @ desc1.transpose(0, 1)

        mask = (matches == matches.max(dim=1, keepdim=True).values) & \
               (matches == matches.max(dim=0, keepdim=True).values)
        valid, indices = mask.max(dim=1)
        ratio = torch.topk(matches, k=2, dim=1).values
        # noinspection PyUnresolvedReferences
        ratio = (-2 * ratio + 2).sqrt()
        ratio = (ratio[:, 0] / ratio[:, 1]) < 0.8
        valid = valid & ratio

        kpts0 = kpts0[valid] * data['scale0']
        kpts1 = kpts1[indices[valid]] * data['scale1']
        mconf = matches.max(dim=1).values[valid]

        b_ids = torch.where(valid[None])[0]

        data.update({
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
            'mkpts0_f': kpts0,
            'mkpts1_f': kpts1,
            'm_bids': b_ids,
            'mconf': mconf,
        })

    def test_step(self, batch, batch_idx):
        self.inference(batch)
        metrics = self.compute_metrics(batch)
        return {'Metrics': metrics}

    def test_epoch_end(self, outputs):

        metrics = [o['Metrics'] for o in outputs]
        metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in metrics]))) for k in metrics[0]}

        unq_ids = list(OrderedDict((iden, i) for i, iden in enumerate(metrics['identifiers'])).values())
        ord_ids = sorted(unq_ids, key=lambda x:metrics['identifiers'][x])
        metrics = {k:[v[x] for x in ord_ids] for k,v in metrics.items()}
        # ['identifiers', 'epi_errs', 'R_errs', 't_errs', 'inliers',
        #  'covisible0', 'covisible1', 'Rot', 'Tns', 'Rot1', 'Tns1']
        output = ''
        output += 'identifiers covisible0 covisible1 R_errs t_errs t_errs2 '
        output += 'Bef.Prec Bef.Num Aft.Prec Aft.Num\n'
        eet = 5e-4  # epi_err_thr
        mean = lambda x: sum(x) / max(len(x), 1)
        for ids, epi, Rer, Ter, Ter2, inl, co0, co1 in zip(
                metrics['identifiers'], metrics['epi_errs'],
                metrics['R_errs'], metrics['t_errs'], metrics['t_errs2'], metrics['inliers'],
                metrics['covisible0'], metrics['covisible1']):
            bef = epi < eet
            aft = epi[inl] < eet
            output += f'{ids} {co0} {co1} {Rer} {Ter} {Ter2} '
            output += f'{mean(bef)} {sum(bef)} {mean(aft)} {sum(aft)}\n'

        scene = Path(self.hparams['dcfg'][self.pcfg["tests"]]['DATASET']['TESTS']['LIST_PATH']).stem.split('_')[0]
        path = f"dump/zeb/[T] {self.pcfg.weight} {scene:>15} {self.pcfg.version}.txt"
        with open(path, 'w') as file:
            file.write(output)
