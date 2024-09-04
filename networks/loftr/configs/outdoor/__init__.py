from networks.loftr.config import get_cfg_defaults as get_network_cfg
from trainer.config import get_cfg_defaults as get_trainer_cfg

# network
network_cfg = get_network_cfg()
network_cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

# optimizer
trainer_cfg = get_trainer_cfg()
trainer_cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
trainer_cfg.TRAINER.WARMUP_RATIO = 0.1
trainer_cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
