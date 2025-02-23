# -*- coding: utf-8 -*-
# @Author  : xuelun
import os
import math
import warnings
import pytorch_lightning as pl

from os.path import join
from argparse import ArgumentParser
from yacs.config import CfgNode as CN
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger

import tools as com

from trainer import Trainer
from modules.config import get_cfg_defaults as get_network_cfg
from trainer.config import get_cfg_defaults as get_trainer_cfg
from trainer.debug import get_cfg_defaults as get_debug_cfg

from datasets.data import MultiSceneDataModule
from datasets import megadepth
from datasets import scannet
from datasets import walk

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

Benchmarks = dict(
    MegaDepth     = megadepth.cfg,
    ScanNet       = scannet.cfg,
    WALK          = walk.cfg,
)


if __name__ == '__main__':
    # ------------
    # Hyperparameters
    # ------------
    parser = ArgumentParser()

    # Project args
    parser.add_argument('--trains', type=str, choices=set(Benchmarks), nargs='+',
                        default=['MegaDepth', 'ScanNet', 'WALK'],
                        help=f'Train Datasets: {set(Benchmarks)}', )
    parser.add_argument('--valids', type=str, choices=set(Benchmarks), nargs='+',
                        default=['MegaDepth'],
                        help=f'Valid Datasets: {set(Benchmarks)}', )
    parser.add_argument('--tests', type=str, choices=set(Benchmarks),
                        default=None,
                        help=f'Test Datasets: {set(Benchmarks)}', )
    parser.add_argument('--debug', action='store_true',
                        help='For debug mode')

    # Loader args
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training and validation (default=2)')
    parser.add_argument('--valid_batch_size', type=int, default=2,
                        help='input batch size for training and validation (default=2)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads (default: 8)')

    # Traner args
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU numbers')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='Cluster node numbers')
    parser.add_argument('--max_epochs', type=int, required=True,
                        help='Traning epochs')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Learning Rate Warm-up steps')
    parser.add_argument("--git", type=str, default='xxxxxx',
                        help=f'Git ID',)
    parser.add_argument("--wid", type=str, default='xxxxxx',
                        help=f'Run ID',)
    parser.add_argument("--weight", type=str, default=None,
                        help=f'Pretrained model weight',)

    # Hyper-parameters
    parser.add_argument('--img_size', type=int, nargs='+', default=[640, 640],
                        help='Image Size: [width, height]')
    parser.add_argument('--lr', type=float, default=4e-3,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='Minimal Learning rate')

    # Runtime args
    parser.add_argument('--test', action='store_true',
                        help="Tesing")
    parser.add_argument('--viz', action='store_true',
                        help="Tesing")

    parser.add_argument('--maxlen', type=int, nargs='+',
                        required=True,
                        help='Accumulate Grad Batches')
    parser.add_argument('--resample', action='store_true',
                        help='Resample training data each epoch')

    args = parser.parse_args()

    # ------------
    # Project config
    # ------------
    pcfg = CN(vars(args))
    tcfg = get_trainer_cfg()
    ncfg = get_network_cfg()
    dcfg = CN({x:Benchmarks.get(x, None) for x in set(args.trains + args.valids + [args.tests])})
    if args.debug: tcfg.merge_from_other_cfg(get_debug_cfg())
    dcfg.DF = ncfg.LOFTR.RESOLUTION[0]
    ncfg.LOFTR.WEIGHT = args.weight

    # load weight
    if args.weight is not None:
        ckpt_path = com.find_in_dir(args.weight, 'wandb')
        ckpt_path = join('wandb', ckpt_path, 'files', 'checkpoints')
        weight = com.ckpt_in_dir('AUC', ckpt_path)
        weight = join(ckpt_path, weight)
        ncfg.LOFTR.WEIGHT = weight

    # ------------
    # Update Trainer Config
    # ------------
    TRAINER = tcfg.TRAINER
    TRAINER.TRUE_BATCH_SIZE = args.gpus * args.batch_size * args.num_nodes
    pcfg.accumulate_grad_batches = math.ceil(TRAINER.CANONICAL_BS / TRAINER.TRUE_BATCH_SIZE)
    TRAINER.TRUE_LR = TRAINER.CANONICAL_LR = args.lr
    TRAINER.NUM_SANITY_VAL_STEPS = 0 if args.weight is None or args.debug else -1

    # ------------
    # W&B logger
    # ------------
    wid = args.wid
    com.hint('ID = {}'.format(wid))
    logger = TensorBoardLogger('tensorboard', name=args.git, version=wid)

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
    valid_nums = 1  # validation_nums_each_epoch
    checkpoint_path = f'tensorboard/{args.git}/{args.wid}/checkpoints/last.ckpt'
    if os.path.isfile(checkpoint_path):
        resume_checkpoint = checkpoint_path
    else:
        resume_checkpoint = None
    fitter = pl.Trainer.from_argparse_args(
        args,
        # ddp
        sync_batchnorm=True,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=args.resample,  # avoid repeated samples!
        strategy=DDPPlugin(find_unused_parameters=False),
        # reproducible
        benchmark=True,
        deterministic=False,
        # logger
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=TRAINER.LOG_INTERVAL,
        # prepare
        val_check_interval=1/valid_nums,
        num_sanity_val_steps=TRAINER.NUM_SANITY_VAL_STEPS,
        limit_train_batches=TRAINER.LIMIT_TRAIN_BATCHES,
        limit_val_batches=TRAINER.LIMIT_VALID_BATCHES,
        # faster training
        accumulate_grad_batches=pcfg.accumulate_grad_batches,
        # amp_level=TRAINER.AMP_LEVEL,
        # amp_backend=TRAINER.AMP_BACKEND,
        # precision=TRAINER.PRECISION,  #https://github.com/PyTorchLightning/pytorch-lightning/issues/5558
        # better fine-tune
        gradient_clip_val=TRAINER.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm=TRAINER.GRADIENT_CLIP_ALGORITHM,
        resume_from_checkpoint=resume_checkpoint,
    )

    # ------------
    # Fitting
    # ------------
    if args.test:
        fitter.test(trainer, datamodule=dm)
    else:
        fitter.fit(trainer, datamodule=dm)
