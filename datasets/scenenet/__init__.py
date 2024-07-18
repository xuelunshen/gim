# -*- coding: utf-8 -*-
# @Author  : xuelun

from os.path import join
from yacs.config import CfgNode as CN

##########################################
#++++++++++++++++++++++++++++++++++++++++#
#+                                      +#
#+            SceneNet-RGBD             +#
#+                                      +#
#++++++++++++++++++++++++++++++++++++++++#
##########################################

_CN = CN()

_CN.DATASET = CN()

DATA_ROOT = 'data/SceneNetRGBD/'
NPZ_ROOT = DATA_ROOT

_CN.NJOBS = 1

# TRAIN
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.PADDING = None
_CN.DATASET.TRAIN.DATA_ROOT = None
_CN.DATASET.TRAIN.NPZ_ROOT = None
_CN.DATASET.TRAIN.MAX_SAMPLES = None
_CN.DATASET.TRAIN.MIN_OVERLAP_SCORE = None
_CN.DATASET.TRAIN.MAX_OVERLAP_SCORE = None
_CN.DATASET.TRAIN.AUGMENTATION_TYPE = None
_CN.DATASET.TRAIN.LIST_PATH = None

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
_CN.DATASET.TESTS.PADDING = False
_CN.DATASET.TESTS.DATA_ROOT = join(DATA_ROOT, 'test')
_CN.DATASET.TESTS.NPZ_ROOT = NPZ_ROOT
_CN.DATASET.TESTS.MAX_SAMPLES = 30
_CN.DATASET.TESTS.MIN_OVERLAP_SCORE = 0.0
_CN.DATASET.TESTS.MAX_OVERLAP_SCORE = 0.5
_CN.DATASET.TESTS.AUGMENTATION_TYPE = None
_CN.DATASET.TESTS.LIST_PATH = 'datasets/_tests_/SceneNetRGBD.txt'

cfg = _CN
