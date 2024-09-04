# -*- coding: utf-8 -*-
# @Author  : xuelun

from yacs.config import CfgNode as CN

_CN = CN()

# ------------
# Trainer
# ------------
_CN.TRAINER = CN()
_CN.TRAINER.SEED = 3407
_CN.TRAINER.NUM_SANITY_VAL_STEPS = -1
_CN.TRAINER.LOG_INTERVAL = 20
_CN.TRAINER.VAL_CHECK_INTERVAL = 1.0 # default 1.0, if we set 2.0 will val each 2 step
_CN.TRAINER.LIMIT_TRAIN_BATCHES = 1.0 # default 1.0
_CN.TRAINER.LIMIT_VALID_BATCHES = 1.0 # default 1.0 will use all training batch
_CN.TRAINER.AMP_LEVEL = 'O1' # 'O1' for apex
_CN.TRAINER.AMP_BACKEND = 'apex' # 'O1' for apex
_CN.TRAINER.PRECISION = 16 # default 32
_CN.TRAINER.GRADIENT_CLIP_VAL = 0.5 # default 0.0
_CN.TRAINER.GRADIENT_CLIP_ALGORITHM = 'norm' # default 'norm'

# optimizer
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1
# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800
# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval
# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = None
_CN.TRAINER.RANSAC_CONF = 0.999999
_CN.TRAINER.RANSAC_MAX_ITERS = 100000
_CN.TRAINER.USE_MAGSACPP = False

# Related to Visualization
_CN.VISUAL = CN()
_CN.VISUAL.N_VAL_PAIRS_TO_PLOT = 10
_CN.VISUAL.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.VISUAL.PLOT_MATCHES_ALPHA = 'dynamic'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
