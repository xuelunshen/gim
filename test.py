# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import math
import uuid

import pytorch_lightning as pl

from pathlib import Path
from os.path import join, exists
from argparse import ArgumentParser
from yacs.config import CfgNode as CN
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger

import tools as com

from trainer import Trainer
from networks.loftr.configs.outdoor import trainer_cfg, network_cfg
from networks.loftr.config import get_cfg_defaults as get_network_cfg
from trainer.config import get_cfg_defaults as get_trainer_cfg
from trainer.debug import get_cfg_defaults as get_debug_cfg

from datasets.data import MultiSceneDataModule
from datasets import gl3d
from datasets import gtasfm
from datasets import multifov
from datasets import blendedmvs
from datasets import iclnuim
from datasets import scenenet
from datasets import eth3d
from datasets import kitti
from datasets import robotcar

Benchmarks = dict(
    GL3D            = gl3d.cfg,
    GTASfM          = gtasfm.cfg,
    MultiFoV        = multifov.cfg,
    BlendedMVS      = blendedmvs.cfg,
    ICLNUIM         = iclnuim.cfg,
    SceneNet        = scenenet.cfg,
    ETH3DO          = eth3d.cfgO,
    ETH3DI          = eth3d.cfgI,
    KITTI           = kitti.cfg,
    RobotcarNight   = robotcar.night,
    RobotcarSeason  = robotcar.season,
    RobotcarWeather = robotcar.weather,
)

RANSACs = dict(
    RANSAC = cv2.RANSAC,
    FAST = cv2.USAC_FAST,
    MAGSAC = cv2.USAC_MAGSAC,
    PROSAC = cv2.USAC_PROSAC,
    DEFAULT = cv2.USAC_DEFAULT,
    ACCURATE = cv2.USAC_ACCURATE,
    PARALLEL = cv2.USAC_PARALLEL,
)

MODEL_ZOO = ['gim_dkm', 'gim_loftr', 'gim_lightglue', 'root_sift']


if __name__ == '__main__':
    # ------------
    # Hyperparameters
    # ------------
    parser = ArgumentParser()

    # Project args
    parser.add_argument('--trains', type=str, choices=set(Benchmarks), nargs='+',
                        default=[],
                        help=f'Train Datasets: {set(Benchmarks)}', )
    parser.add_argument('--valids', type=str, choices=set(Benchmarks), nargs='+',
                        default=[],
                        help=f'Valid Datasets: {set(Benchmarks)}', )
    parser.add_argument('--tests', type=str, choices=set(Benchmarks),
                        default=None,
                        help=f'Test Datasets: {set(Benchmarks)}', )
    parser.add_argument('--debug', action='store_true',
                        help='For debug mode')

    # Loader args
    parser.add_argument('--batch_size', type=int, default=12,
                        help='input batch size for training and validation (default=2)')
    parser.add_argument('--threads', type=int, default=3,
                        help='Number of threads (default: 3)')

    # Traner args
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU numbers')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Cluster node numbers')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Traning epochs (default: 30)')
    parser.add_argument("--git", type=str, default='xxxxxx',
                        help=f'Git ID',)
    parser.add_argument("--weight", type=str, default=None, choices=MODEL_ZOO,
                        required=True,
                        help=f'Pretrained model weight',)

    # Hyper-parameters
    parser.add_argument('--img_size', type=int, default=9999,
                        help='Image Size')
    parser.add_argument('--lr', type=float, default=8e-3,
                        help='Learning rate')

    # Runtime args
    parser.add_argument('--test', action='store_true',
                        help="Tesing")
    parser.add_argument('--viz', action='store_true',
                        help="Tesing")

    parser.add_argument("--max_samples", type=int, default=None,
                        help=f'Max Samples in Testing',)
    parser.add_argument("--min_score", type=float, default=0.0,
                        help='Min Score in Testing',)
    parser.add_argument("--max_score", type=float, default=1.0,
                        help='Max Score in Testing',)

    parser.add_argument("--ransac_threshold", type=float, default=0.5,
                        help='RANSAC Threshold',)
    parser.add_argument('--ransac', type=str, choices=set(RANSACs), default='MAGSAC',
                        help=f'RANSAC Methods: {set(RANSACs)}', )
    parser.add_argument("--version", type=str, default='AUC',
                        help=f'Model version',)

    args = parser.parse_args()

    # ------------
    # Project config
    # ------------
    pcfg = CN(vars(args))
    tcfg = get_trainer_cfg()
    ncfg = get_network_cfg()
    dcfg = CN({x:Benchmarks.get(x, None) for x in set(args.trains + args.valids + [args.tests])})
    tcfg.merge_from_other_cfg(trainer_cfg)
    if args.debug: tcfg.merge_from_other_cfg(get_debug_cfg())
    ncfg.merge_from_other_cfg(network_cfg)
    dcfg.DF = ncfg.LOFTR.RESOLUTION[0]

    # load weight
    ncfg.LOFTR.WEIGHT = join('weights', args.weight + '_' + args.version + '.ckpt')
    if args.weight == 'root_sift':
        ncfg.LOFTR.WEIGHT = None

    # ------------
    # Testing setting
    # ------------
    if args.max_samples is not None and args.test: dcfg[args.tests]['DATASET']['TESTS']['MAX_SAMPLES'] = args.max_samples
    if args.min_score is not None and args.test: dcfg[args.tests]['DATASET']['TESTS']['MIN_OVERLAP_SCORE'] = args.min_score
    if args.max_score is not None and args.test: dcfg[args.tests]['DATASET']['TESTS']['MAX_OVERLAP_SCORE'] = args.max_score
    # print(dcfg)

    # ------------
    # Update Trainer Config
    # ------------
    TRAINER = tcfg.TRAINER
    TRAINER.TRUE_BATCH_SIZE = args.gpus * args.batch_size
    TRAINER.SCALING = _scaling = TRAINER.TRUE_BATCH_SIZE / TRAINER.CANONICAL_BS
    TRAINER.CANONICAL_LR = args.lr
    TRAINER.TRUE_LR = TRAINER.CANONICAL_LR * _scaling
    TRAINER.WARMUP_STEP = math.floor(TRAINER.WARMUP_STEP / _scaling)
    TRAINER.RANSAC_PIXEL_THR = args.ransac_threshold
    TRAINER.POSE_ESTIMATION_METHOD = RANSACs[args.ransac]

    # ------------
    # W&B logger
    # ------------
    # com.login(args.server)
    wid = str(uuid.uuid1()).split('-')[0]
    com.hint('ID = {}'.format(wid))
    logger = TensorBoardLogger('tensorboard', name='test', version='test')

    # ------------
    # reproducible
    # ------------
    pl.seed_everything(TRAINER.SEED, workers=True)
    
    # ------------
    # data loader
    # ------------
    dm = MultiSceneDataModule(args, dcfg)

    # ------------
    # model
    # ------------
    trainer = Trainer(pcfg, tcfg, dcfg, ncfg)
    
    # ------------
    # training
    # ------------
    fitter = pl.Trainer.from_argparse_args(
        args,
        # ddp
        sync_batchnorm=True,
        strategy=DDPPlugin(find_unused_parameters=False),
        # reproducible
        benchmark=True,
        deterministic=False,
        # logger
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=TRAINER.LOG_INTERVAL,
        # prepare
        weights_summary='top',
        val_check_interval=TRAINER.VAL_CHECK_INTERVAL,
        num_sanity_val_steps=TRAINER.NUM_SANITY_VAL_STEPS,
        limit_train_batches=TRAINER.LIMIT_TRAIN_BATCHES,
        limit_val_batches=TRAINER.LIMIT_VALID_BATCHES,
        # faster training
        # amp_level=TRAINER.AMP_LEVEL,
        # amp_backend=TRAINER.AMP_BACKEND,
        # precision=TRAINER.PRECISION,  #https://github.com/PyTorchLightning/pytorch-lightning/issues/5558
        # better fine-tune
        gradient_clip_val=TRAINER.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm=TRAINER.GRADIENT_CLIP_ALGORITHM,
    )

    # ------------
    # Fitting
    # ------------
    if args.test:
        scene = Path(dcfg[pcfg["tests"]]['DATASET']['TESTS']['LIST_PATH']).stem.split('_')[0]
        path = f"dump/zeb/[T] {pcfg.weight} {scene:>15} {pcfg.version}.txt"
        if exists(path):
            print(f"{path} already exists")
            exit(0)
        elif not exists(str(Path(path).parent)):
            Path(path).parent.mkdir(parents=True)
        fitter.test(trainer, datamodule=dm)
    else:
        fitter.fit(trainer, datamodule=dm)
