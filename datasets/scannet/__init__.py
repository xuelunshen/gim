# -*- coding: utf-8 -*-
# @Author  : xuelun

from os.path import join
from yacs.config import CfgNode as CN

##########################################
#++++++++++++++++++++++++++++++++++++++++#
#+                                      +#
#+               ScanNet                +#
#+                                      +#
#++++++++++++++++++++++++++++++++++++++++#
##########################################

_CN = CN()

_CN.DATASET = CN()

DATA_ROOT = 'data/ScanNet/'
NPZ_ROOT = DATA_ROOT

_CN.NJOBS = 8  # 1513 scenes

# TRAIN
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.PADDING = True
_CN.DATASET.TRAIN.DATA_ROOT = join(DATA_ROOT, 'train')
_CN.DATASET.TRAIN.NPZ_ROOT = join(NPZ_ROOT, 'train_scene_info', 'npz')
_CN.DATASET.TRAIN.MAX_SAMPLES = 10000
_CN.DATASET.TRAIN.MIN_OVERLAP_SCORE = 0.0
_CN.DATASET.TRAIN.MAX_OVERLAP_SCORE = 0.7
_CN.DATASET.TRAIN.AUGMENTATION_TYPE = None
_CN.DATASET.TRAIN.LIST_PATH = 'datasets/_train_/ScanNet.txt'

# VALID
_CN.DATASET.VALID = CN()
_CN.DATASET.VALID.PADDING = True
_CN.DATASET.VALID.DATA_ROOT = join(DATA_ROOT, 'test')
_CN.DATASET.VALID.NPZ_ROOT = join(NPZ_ROOT, 'test_scene_info', 'npz')
_CN.DATASET.VALID.MAX_SAMPLES = 20
_CN.DATASET.VALID.MIN_OVERLAP_SCORE = 0.2
_CN.DATASET.VALID.MAX_OVERLAP_SCORE = 0.4
_CN.DATASET.VALID.AUGMENTATION_TYPE = None
_CN.DATASET.VALID.LIST_PATH = None

# TESTS
_CN.DATASET.TESTS = CN()
_CN.DATASET.TESTS.PADDING = False
_CN.DATASET.TESTS.DATA_ROOT = join(DATA_ROOT, 'test')
_CN.DATASET.TESTS.NPZ_ROOT = join(NPZ_ROOT, 'test_scene_info', 'npz')
_CN.DATASET.TESTS.MAX_SAMPLES = 51
_CN.DATASET.TESTS.MIN_OVERLAP_SCORE = 0.0
_CN.DATASET.TESTS.MAX_OVERLAP_SCORE = 0.5
_CN.DATASET.TESTS.AUGMENTATION_TYPE = None
_CN.DATASET.TESTS.LIST_PATH = None

# OTHERS
_CN.DATASET.TRAIN.INTRINSIC_PATH = join(NPZ_ROOT, 'train_scene_info', 'intrinsics.npz')
_CN.DATASET.VALID.INTRINSIC_PATH = join(NPZ_ROOT, 'test_scene_info', 'intrinsics.npz')
_CN.DATASET.TESTS.INTRINSIC_PATH = join(NPZ_ROOT, 'test_scene_info', 'intrinsics.npz')

cfg = _CN
