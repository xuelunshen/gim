# -*- coding: utf-8 -*-
# @Author  : xuelun

from os.path import join
from yacs.config import CfgNode as CN

##########################################
#++++++++++++++++++++++++++++++++++++++++#
#+                                      +#
#+              MegaDepth               +#
#+                                      +#
#++++++++++++++++++++++++++++++++++++++++#
##########################################

_CN = CN()

_CN.DATASET = CN()

DATA_ROOT = 'data/MegaDepth/'
NPZ_ROOT = join(DATA_ROOT, 'output')

_CN.NJOBS = 8  # 153  scenes

# TRAIN
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.PADDING = True
_CN.DATASET.TRAIN.DATA_ROOT = DATA_ROOT
_CN.DATASET.TRAIN.NPZ_ROOT = NPZ_ROOT
_CN.DATASET.TRAIN.MAX_SAMPLES = 100000
_CN.DATASET.TRAIN.MIN_OVERLAP_SCORE = 0.01
_CN.DATASET.TRAIN.MAX_OVERLAP_SCORE = 0.7
_CN.DATASET.TRAIN.AUGMENTATION_TYPE = 'dark'
_CN.DATASET.TRAIN.LIST_PATH = 'datasets/_train_/MegaDepth.txt'

# VALID
_CN.DATASET.VALID = CN()
_CN.DATASET.VALID.PADDING = True
_CN.DATASET.VALID.DATA_ROOT = DATA_ROOT
_CN.DATASET.VALID.NPZ_ROOT = NPZ_ROOT
_CN.DATASET.VALID.MAX_SAMPLES = 2000
_CN.DATASET.VALID.MIN_OVERLAP_SCORE = 0.1
_CN.DATASET.VALID.MAX_OVERLAP_SCORE = 0.7
_CN.DATASET.VALID.AUGMENTATION_TYPE = None
_CN.DATASET.VALID.LIST_PATH = 'datasets/_train_/MegaDepthVal.txt'

# TESTS
_CN.DATASET.TESTS = CN()
_CN.DATASET.TESTS.PADDING = False
_CN.DATASET.TESTS.DATA_ROOT = DATA_ROOT
_CN.DATASET.TESTS.NPZ_ROOT = NPZ_ROOT
_CN.DATASET.TESTS.MAX_SAMPLES = 100
_CN.DATASET.TESTS.MIN_OVERLAP_SCORE = 0.0
_CN.DATASET.TESTS.MAX_OVERLAP_SCORE = 0.5
_CN.DATASET.TESTS.AUGMENTATION_TYPE = None
_CN.DATASET.TESTS.LIST_PATH = None

cfg = _CN
