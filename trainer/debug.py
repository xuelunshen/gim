# -*- coding: utf-8 -*-
# @Author  : xuelun

from yacs.config import CfgNode as CN

_CN = CN()

# ------------
# Trainer
# ------------
_CN.TRAINER = CN()
_CN.TRAINER.NUM_SANITY_VAL_STEPS = 0
_CN.TRAINER.LOG_INTERVAL = 1
_CN.TRAINER.VAL_CHECK_INTERVAL = 1.0 # default 1.0, if we set 2.0 will val each 2 step
_CN.TRAINER.LIMIT_TRAIN_BATCHES = 10.0 # default 1.0
_CN.TRAINER.LIMIT_VALID_BATCHES = 10.0 # default 1.0 will use all training batch


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
