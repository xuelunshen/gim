# -*- coding: utf-8 -*-
# @Author  : xuelun

from os.path import join
from yacs.config import CfgNode as CN

##########################################
#++++++++++++++++++++++++++++++++++++++++#
#+                                      +#
#+                 WALK                 +#
#+                                      +#
#++++++++++++++++++++++++++++++++++++++++#
##########################################

_CN = CN()

_CN.DATASET = CN()

DATA_ROOT = join('data', 'ZeroMatch')
NPZ_ROOT = join(DATA_ROOT, 'pseudo')

_CN.NJOBS = 1  # x scenes

# TRAIN
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.PADDING = True
_CN.DATASET.TRAIN.DATA_ROOT = join(DATA_ROOT, 'video_1080p')
_CN.DATASET.TRAIN.NPZ_ROOT = NPZ_ROOT
_CN.DATASET.TRAIN.MAX_SAMPLES = -1
_CN.DATASET.TRAIN.MIN_OVERLAP_SCORE = None
_CN.DATASET.TRAIN.MAX_OVERLAP_SCORE = None
_CN.DATASET.TRAIN.AUGMENTATION_TYPE = 'dark'
_CN.DATASET.TRAIN.LIST_PATH = 'datasets/_train_/100h.txt'

# OTHERS
_CN.DATASET.TRAIN.STEP = 1000
_CN.DATASET.TRAIN.PIX_THR = 1
_CN.DATASET.TRAIN.MAX_CANDIDATE_MATCHES = -1
_CN.DATASET.TRAIN.MIN_FINAL_MATCHES = 512
_CN.DATASET.TRAIN.MIN_FILTER_MATCHES = 32
_CN.DATASET.TRAIN.FIX_MATCHES = 100000
_CN.DATASET.TRAIN.SOURCE_ROOT = join(DATA_ROOT, 'video_1080p')
_CN.DATASET.TRAIN.PROPAGATE_ROOT = join(DATA_ROOT, 'propagate')
_CN.DATASET.TRAIN.VIDEO_IMAGE_ROOT = join(DATA_ROOT, 'image_1080p')
_CN.DATASET.TRAIN.PSEUDO_LABELS = [
    'WALK SIFT [R] F [S] 10',
    'WALK SIFT [R] F [S] 20',
    'WALK SIFT [R] F [S] 40',
    'WALK SIFT [R] F [S] 80',
    'WALK SIFT [R] T [S] 10',
    'WALK SIFT [R] T [S] 20',
    'WALK SIFT [R] T [S] 40',
    'WALK SIFT [R] T [S] 80',

    'WALK GIM_DKM [R] F [S] 10',
    'WALK GIM_DKM [R] F [S] 20',
    'WALK GIM_DKM [R] F [S] 40',
    'WALK GIM_DKM [R] F [S] 80',
    'WALK GIM_DKM [R] T [S] 10',
    'WALK GIM_DKM [R] T [S] 20',
    'WALK GIM_DKM [R] T [S] 40',
    'WALK GIM_DKM [R] T [S] 80',

    'WALK GIM_GLUE [R] F [S] 10',
    'WALK GIM_GLUE [R] F [S] 20',
    'WALK GIM_GLUE [R] F [S] 40',
    'WALK GIM_GLUE [R] F [S] 80',
    'WALK GIM_GLUE [R] T [S] 10',
    'WALK GIM_GLUE [R] T [S] 20',
    'WALK GIM_GLUE [R] T [S] 40',
    'WALK GIM_GLUE [R] T [S] 80',

    'WALK GIM_LOFTR [R] F [S] 10',
    'WALK GIM_LOFTR [R] F [S] 20',
    'WALK GIM_LOFTR [R] F [S] 40',
    'WALK GIM_LOFTR [R] F [S] 80',
    'WALK GIM_LOFTR [R] T [S] 10',
    'WALK GIM_LOFTR [R] T [S] 20',
    'WALK GIM_LOFTR [R] T [S] 40',
    'WALK GIM_LOFTR [R] T [S] 80',
]

# VALID
_CN.DATASET.VALID = CN()
_CN.DATASET.VALID.PADDING = None
_CN.DATASET.VALID.DATA_ROOT = None
_CN.DATASET.VALID.NPZ_ROOT = None
_CN.DATASET.VALID.MAX_SAMPLES = None
_CN.DATASET.VALID.MIN_OVERLAP_SCORE = None
_CN.DATASET.VALID.MAX_OVERLAP_SCORE = None
_CN.DATASET.VALID.AUGMENTATION_TYPE = None
_CN.DATASET.VALID.LIST_PATH = None

# TESTS
_CN.DATASET.TESTS = CN()
_CN.DATASET.TESTS.PADDING = None
_CN.DATASET.TESTS.DATA_ROOT = None
_CN.DATASET.TESTS.NPZ_ROOT = None
_CN.DATASET.TESTS.MAX_SAMPLES = None
_CN.DATASET.TESTS.MIN_OVERLAP_SCORE = None
_CN.DATASET.TESTS.MAX_OVERLAP_SCORE = None
_CN.DATASET.TESTS.AUGMENTATION_TYPE = None
_CN.DATASET.TESTS.LIST_PATH = None

cfg = _CN
