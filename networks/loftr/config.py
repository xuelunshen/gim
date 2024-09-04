from yacs.config import CfgNode as CN

_CN = CN()
_CN.TEMP_BUG_FIX = True

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR = CN()
_CN.LOFTR.WEIGHT = None

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR.BACKBONE_TYPE = 'ResNetFPN'
_CN.LOFTR.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.LOFTR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.LOFTR.FINE_CONCAT_COARSE_FEAT = False

# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR.RESNETFPN = CN()
_CN.LOFTR.RESNETFPN.INITIAL_DIM = 128
_CN.LOFTR.RESNETFPN.BLOCK_DIMS = [64, 128, 196, 256, 512, 1024]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 256
_CN.LOFTR.COARSE.NHEAD = 8
_CN.LOFTR.COARSE.LAYER_NAMES = 4
_CN.LOFTR.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']

# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.2
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 2
_CN.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTR.MATCH_COARSE.SKH_ITERS = 3
_CN.LOFTR.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.LOFTR.MATCH_COARSE.SKH_PREFILTER = False
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

# 4. LoFTR-fine module config
_CN.LOFTR.FINE = CN()
_CN.LOFTR.FINE.D_MODEL = 128
_CN.LOFTR.FINE.NHEAD = 8
_CN.LOFTR.FINE.LAYER_NAMES = 1
_CN.LOFTR.FINE.ATTENTION = 'linear'

# 5. LoFTR Losses
# -- # coarse-level
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
# _CN.LOFTR.LOSS.SPARSE_SPVS = False
# -- - -- # focal loss (coarse)
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0
# _CN.LOFTR.LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# use `_CN.LOFTR.MATCH_COARSE.MATCH_TYPE`

# -- # fine-level
_CN.LOFTR.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

# Overlap
_CN.LOFTR.LOSS.OVERLAP_WEIGHT = 20.0
_CN.LOFTR.LOSS.OVERLAP_FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.OVERLAP_FOCAL_GAMMA = 5.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
