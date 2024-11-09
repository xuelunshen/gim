# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import torch
import warnings
import numpy as np

from tqdm import tqdm
from os.path import join
from pathlib import Path
from argparse import ArgumentParser

from hloc import pairs_from_exhaustive
from hloc import extract_features, match_features, match_dense, reconstruction

from hloc.utils import segment
from hloc.utils.io import read_image
from hloc.match_dense import ImagePairDataset

from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue
from networks.mit_semseg.models import ModelBuilder, SegmentationModule


def segmentation(images, segment_root, matcher_conf):
    # initial device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # initial segmentation mode
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
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module = segmentation_module.to(device).eval()
    # initial data reader
    dataset = ImagePairDataset(None, matcher_conf["preprocessing"], None)
    # Segment images
    image_list = sorted(os.listdir(images))
    with torch.no_grad():
        for img in tqdm(image_list):
            segment_path = join(segment_root, '{}.npy'.format(img[:-4]))
            if not os.path.exists(segment_path):
                rgb = read_image(images / img, dataset.conf.grayscale)
                mask = segment(rgb, 1920, device, segmentation_module)
                np.save(segment_path, mask)


def main(scene_name, version):
    # Setup
    images = Path('inputs') / scene_name / 'images'

    outputs = Path('outputs') / scene_name / version
    outputs.mkdir(parents=True, exist_ok=True)
    os.environ['GIMRECONSTRUCTION'] = str(outputs)

    segment_root = Path('outputs') / scene_name / 'segment'
    segment_root.mkdir(parents=True, exist_ok=True)

    sfm_dir = outputs / 'sparse'
    mvs_path = outputs / 'dense'
    database_path = sfm_dir / 'database.db'
    image_pairs = outputs / 'pairs-near.txt'

    feature_conf = matcher_conf = None

    if version == 'gim_dkm':
        feature_conf = None
        matcher_conf = match_dense.confs[version]
    elif version == 'gim_lightglue':
        feature_conf = extract_features.confs['gim_superpoint']
        matcher_conf = match_features.confs[version]
    
    # Find image pairs via pair-wise image
    exhaustive_pairs = pairs_from_exhaustive.main(image_pairs, image_list=images)

    segmentation(images, segment_root, matcher_conf)

    # Extract and match local features
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if version == 'gim_dkm':
            feature_path, match_path = match_dense.main(matcher_conf, image_pairs,
                                                        images, outputs)
        elif version == 'gim_lightglue':
            checkpoints_path = join('weights', 'gim_lightglue_100h.ckpt')

            detector = SuperPoint({
                'max_num_keypoints': 2048,
                'force_num_keypoints': True,
                'detection_threshold': 0.0,
                'nms_radius': 3,
                'trainable': False,
            })
            state_dict = torch.load(checkpoints_path, map_location='cpu')
            if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('model.'):
                    state_dict.pop(k)
                if k.startswith('superpoint.'):
                    state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
            detector.load_state_dict(state_dict)

            model = LightGlue({
                'filter_threshold': 0.1,
                'flash': False,
                'checkpointed': True,
            })
            state_dict = torch.load(checkpoints_path, map_location='cpu')
            if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('superpoint.'):
                    state_dict.pop(k)
                if k.startswith('model.'):
                    state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            model.load_state_dict(state_dict)

            feature_path = extract_features.main(feature_conf, images, outputs,
                                                 model=detector)
            match_path = match_features.main(matcher_conf, image_pairs,
                                             feature_conf['output'], outputs,
                                             model=model)

    # sparse reconstruction
    reconstruction.main(sfm_dir, images, image_pairs, feature_path, match_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--version', type=str, choices={'gim_dkm', 'gim_lightglue'},
                        default='gim_dkm')
    args = parser.parse_args()
    
    main(args.scene_name, args.version)
