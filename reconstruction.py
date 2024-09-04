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
from hloc import match_dense, reconstruction

from hloc.utils import segment
from hloc.utils.io import read_image
from hloc.match_dense import ImagePairDataset

from networks.mit_semseg.models import ModelBuilder, SegmentationModule


def main(scene_name, version):
    # Setup
    images = Path('inputs') / scene_name / 'images'
    
    outputs = Path('outputs') / scene_name / (version+'_dkm')
    outputs.mkdir(parents=True, exist_ok=True)
    os.environ['GIMRECONSTRUCTION'] = str(outputs)
    
    segment_root = outputs / 'segment'
    segment_root.mkdir(parents=True, exist_ok=True)
    
    sfm_dir = outputs / 'sparse'
    mvs_path = outputs / 'dense'
    database_path = sfm_dir / 'database.db'
    image_pairs = outputs / 'pairs-near.txt'

    matcher_conf = match_dense.confs[version]
    
    # Find image pairs via pair-wise image
    exhaustive_pairs = pairs_from_exhaustive.main(image_pairs, image_list=images)
    
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
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit).to(device).eval()
    # initial data reader
    dataset = ImagePairDataset(None, matcher_conf["preprocessing"], None)
    # Segment images
    image_list = sorted(os.listdir(images))
    with torch.no_grad():
        for img in tqdm(image_list):
            rgb = read_image(images / img, dataset.conf.grayscale)
            segment_path = join(segment_root, '{}.npy'.format(img[:-4]))
            if not os.path.exists(segment_path):
                mask = segment(rgb, 1920, device, segmentation_module)
                np.save(segment_path, mask)

    # Extract and match local features
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        feature_path, match_path = match_dense.main(matcher_conf, image_pairs, images,
                                                    outputs, max_kps=8192,
                                                    overwrite=False)
    
    # sparse reconstruction
    reconstruction.main(sfm_dir, images, image_pairs, feature_path, match_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str)
    parser.add_argument('--version', type=str)
    args = parser.parse_args()
    
    main(args.scene_name, args.version)
