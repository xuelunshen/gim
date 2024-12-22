# -*- coding: utf-8 -*-
# @Author  : xuelun

from yacs.config import CfgNode as CN

_CN = CN()

# ------------
# Trainer
# ------------
_CN.TRAINER = CN()
_CN.TRAINER.LOG_INTERVAL = 10


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
