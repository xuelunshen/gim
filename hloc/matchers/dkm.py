import os
import cv2
import torch
import warnings
import numpy as np
from os.path import join
from pathlib import Path

from tools import get_padding_size
from hloc.utils import CLS_DICT, exclude
from ..utils.base_model import BaseModel
from networks.dkm.models.model_zoo.DKMv3 import DKMv3


class LoFTR(BaseModel):
    default_conf = {
        'max_num_matches': None,
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        self.h = 672
        self.w = 896
        model = DKMv3(None, self.h, self.w, upsample_preds=True)

        checkpoints_path = join('weights', conf['weights'])
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)

        self.net = model

    def _forward(self, data):
        outputs = Path(os.environ['GIMRECONSTRUCTION'])
        segment_root = outputs / '..' / 'segment'

        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            'keypoints0': 'keypoints1',
            'keypoints1': 'keypoints0',
            'image0': 'image1',
            'image1': 'image0',
            'mask0': 'mask1',
            'mask1': 'mask0',
            'name0': 'name1',
            'name1': 'name0',
        }
        data_ = {rename[k]: v for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            image0, image1 = data_['image0'], data_['image1']
            img0, img1 = data_['name0'], data_['name1']
            
            # segment image
            seg_path0 = join(segment_root, '{}.npy'.format(img0[:-4]))
            mask0 = np.load(seg_path0)
            if mask0.shape[:2] != image0.shape[-2:]:
                mask0 = cv2.resize(mask0, image0.shape[-2:][::-1],
                                   interpolation=cv2.INTER_NEAREST)
            mask_0 = mask0 != CLS_DICT[exclude[0]]
            for cls in exclude[1:]:
                mask_0 = mask_0 & (mask0 != CLS_DICT[cls])
            mask_0 = mask0
            mask_0 = mask_0.astype(np.uint8)
            mask_0 = torch.from_numpy((mask_0 == 0).astype(np.uint8)).to(image0.device)
            mask_0 = mask_0.float()[None, None] == 0
            image0 = image0 * mask_0
            # segment image
            seg_path1 = join(segment_root, '{}.npy'.format(img1[:-4]))
            mask1 = np.load(seg_path1)
            if mask1.shape != image1.shape[-2:]:
                mask1 = cv2.resize(mask1, image1.shape[-2:][::-1],
                                   interpolation=cv2.INTER_NEAREST)
            mask_1 = mask1 != CLS_DICT[exclude[0]]
            for cls in exclude[1:]:
                mask_1 = mask_1 & (mask1 != CLS_DICT[cls])
            mask_1 = mask1
            mask_1 = mask_1.astype(np.uint8)
            mask_1 = torch.from_numpy((mask_1 == 0).astype(np.uint8)).to(image1.device)
            mask_1 = mask_1.float()[None, None] == 0
            image1 = image1 * mask_1

            orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image0, self.h, self.w)
            orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image1, self.h, self.w)
            image0 = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
            image1 = torch.nn.functional.pad(image1, (pad_left1, pad_right1, pad_top1, pad_bottom1))

            dense_matches, dense_certainty = self.net.match(image0, image1)
            sparse_matches, mconf = self.net.sample(dense_matches, dense_certainty, 8192)

            m = mconf > 0
            mconf = mconf[m]
            sparse_matches = sparse_matches[m]

            height0, width0 = image0.shape[-2:]
            height1, width1 = image1.shape[-2:]

            kpts0 = sparse_matches[:, :2]
            kpts0 = torch.stack((width0 * (kpts0[:, 0] + 1) / 2,
                                 height0 * (kpts0[:, 1] + 1) / 2), dim=-1, )
            kpts1 = sparse_matches[:, 2:]
            kpts1 = torch.stack((width1 * (kpts1[:, 0] + 1) / 2,
                                 height1 * (kpts1[:, 1] + 1) / 2), dim=-1, )
            b_ids, i_ids = torch.where(mconf[None])

            # before padding
            kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
            kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
            mask = (kpts0[:, 0] > 0) & \
                   (kpts0[:, 1] > 0) & \
                   (kpts1[:, 0] > 0) & \
                   (kpts1[:, 1] > 0)
            mask = mask & \
                   (kpts0[:, 0] <= (orig_width0 - 1)) & \
                   (kpts1[:, 0] <= (orig_width1 - 1)) & \
                   (kpts0[:, 1] <= (orig_height0 - 1)) & \
                   (kpts1[:, 1] <= (orig_height1 - 1))

            pred = {
                'keypoints0': kpts0[i_ids],
                'keypoints1': kpts1[i_ids],
                'confidence': mconf[i_ids],
                'batch_indexes': b_ids,
            }

            # noinspection PyUnresolvedReferences
            scores, b_ids = pred['confidence'], pred['batch_indexes']
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            pred['confidence'], pred['batch_indexes'] = scores[mask], b_ids[mask]
            pred['keypoints0'], pred['keypoints1'] = kpts0[mask], kpts1[mask]

        scores = pred['confidence']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['keypoints0'], pred['keypoints1'] =\
                pred['keypoints0'][keep], pred['keypoints1'][keep]
            scores = scores[keep]

        # Switch back indices
        pred = {(rename[k] if k in rename else k): v for k, v in pred.items()}
        pred['scores'] = scores
        del pred['confidence']
        return pred
