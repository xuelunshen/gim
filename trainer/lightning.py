# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import cv2
import math
import torch

import pytorch_lightning as pl

from pathlib import Path
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from modules.loftr import LoFTR
from modules.loss import Loss
from modules.utils.supervision import spvs_coarse, spvs_fine
from tools.comm import all_gather
from tools.metrics import aggregate_metrics
from tools.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors
from tools.misc import lower_config, flattenList
from tools.plotting import fast_make_matching_robust_fitting_figure
from trainer.optimizer import build_optimizer


class Trainer(pl.LightningModule):

    def __init__(self, pcfg, tcfg, dcfg, ncfg):
        super().__init__()

        self.save_hyperparameters()
        self.pcfg = pcfg
        self.tcfg = tcfg
        self.ncfg = ncfg
        ncfg = lower_config(ncfg)
        self.model = LoFTR(ncfg['loftr'])
        self.loss_func = Loss(ncfg)

        self.train_step = 0
        self.valid_step = 0
        self.best = {'AUC@5': 0, 'AUC@10': 0, 'AUC@20': 0, 'Prec@5e-04': 0}
        self.n_vals_plot = tcfg.VISUAL.N_VAL_PAIRS_TO_PLOT
        self.log_every_n_steps = tcfg.TRAINER.LOG_INTERVAL
        self.log_valids = {k: 0 for k in self.pcfg.valids}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_step'] = self.train_step
        checkpoint['valid_step'] = self.valid_step
        checkpoint['EPOCH'] = self.current_epoch + 1
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        self.train_step = checkpoint.get('train_step', 0)
        self.valid_step = checkpoint.get('valid_step', 0)
        os.environ['EPOCH'] = str(checkpoint.get('EPOCH', 0))

    def configure_callbacks(self):
        self.checkpoints_dir = './'
        if self.global_rank == 0:
            self.checkpoints_dir = Path(self.logger.log_dir) / 'checkpoints'
            if not self.pcfg['test']:
                self.checkpoints_dir.mkdir(exist_ok=True, parents=True)

        auc_callback = ModelCheckpoint(monitor='AUC', verbose=False, save_top_k=5,
                                       dirpath=self.checkpoints_dir,
                                       mode='max',
                                       save_last=True,
                                       filename='{epoch}-{AUC:.6f}')
        return [auc_callback, ModelSummary(max_depth=3)]

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.tcfg, key='encode')
        return optimizer

    def learning_rate_step(self):
        min_lr = self.pcfg.min_lr
        TRAINER = self.tcfg.TRAINER
        optimizer = self.trainer.optimizers[0]
        one_epoch_step = math.ceil(len(self.trainer.datamodule.trainer.train_dataloader) / self.pcfg.accumulate_grad_batches)
        all_epoch_step = one_epoch_step * self.pcfg.max_epochs
        warmup_step = one_epoch_step
        if self.trainer.global_step < warmup_step:
            base_lr = TRAINER.WARMUP_RATIO * TRAINER.TRUE_LR
            lr = base_lr + \
                 (self.trainer.global_step / warmup_step) * \
                 abs(TRAINER.TRUE_LR - base_lr)
        else:
            base_lr = all_epoch_step / (all_epoch_step - warmup_step) * (TRAINER.TRUE_LR - min_lr) + min_lr
            remain_step = all_epoch_step - self.trainer.global_step
            lr = remain_step / all_epoch_step * (base_lr - min_lr) + min_lr

        p = 0.1
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = lr*p if i == 0 else lr

    def forward(self, data):
        """
        Args:
            data:
                Note: oH/oW = Original Height and Width
                Note: rH/rW = Resized Height and Width
                Note: * indicate 0 or 1
                'covisible*' {Tensor: (b,)}
                'depth*' {Tensor: (b, oH, oW)}
                'image*' {Tensor: (b, 3, rH, rW)}
                'gray*' {Tensor: (b, 1, oH, oW)}
                'imsize*' {Tensor: (b, 2)} - (oH, oW)
                'K*' {Tensor: (b, 3, 3)}
                'resize*' {Tensor: (b, 2)} - (rH, rW)
                'scale*' {Tensor: (b, 2)} - [oW / rW, oH / rH]
                'T_0to1' {Tensor: (b, 4, 4)}
                'T_1to0' {Tensor: (b, 4, 4)}
                'mask*' {Tensor: (b, rH // df, rW // df)}
                other string: 'dataset_name', 'pair_id', 'pair_names', 'scene_id
        Returns:
            output:
                'prob*': {Tensor: ([b, num_patches])}
        """

        config = self.ncfg['LOFTR']

        if data['gt'].sum(): spvs_coarse(data, config['RESOLUTION'])

        self.model(data)

        if data['gt'].sum(): spvs_fine(data, config['RESOLUTION'], config['FINE_WINDOW_SIZE'])

    def compute_metrics(self, batch):
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch, self.tcfg)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(batch['scene_id'], *batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers'],
            'covisible0': batch['covisible0'],
            'covisible1': batch['covisible1'],
            'dataset_name': batch['dataset_name'],
        }
        return metrics

    def train_log(self, batch_idx, data, details):
        if batch_idx % self.log_every_n_steps == 0:
            # noinspection PyUnresolvedReferences
            dicts = {'ResNet Learning rate': self.optimizers().param_groups[0]['lr'],
                     'Others Learning rate': self.optimizers().param_groups[1]['lr'],
                     'step': self.train_step}
            dicts = {**dicts, **details['loss_scalars']}
            dicts = {'Train/' + k: v.detach().mean() if torch.is_tensor(v) else v for k, v in dicts.items()}
            self.logger.log_metrics(dicts, self.train_step)
            self.train_step += 1

    def training_step(self, batch, batch_idx):
        self.learning_rate_step()
        self.forward(batch)
        details = self.loss_func(batch)
        self.train_log(batch_idx, batch, details)
        return details['loss']

    def training_epoch_end(self, outputs) -> None:
        os.environ['EPOCH'] = str(self.current_epoch + 1)

    def valid_log(self, batch_idx, data, details):
        dicts = dict()
        dicts = {**dicts, **details['loss_scalars']}
        dicts = {'Valid ' + k: v.detach().mean() if torch.is_tensor(v) else v for k, v in dicts.items()}

        return dicts

    def validation_step(self, batch, batch_idx):
        self.forward(batch)
        details = self.loss_func(batch)
        metrics = self.compute_metrics(batch)
        dicts = self.valid_log(batch_idx, batch, details)
        return {'Metrics': metrics, 'Dicts': dicts}

    def aggregate_metrics(self, outputs, test=False):
        metrics = [o['Metrics'] for o in outputs]
        metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in metrics]))) for k in metrics[0] if 'covisible' not in k}
        datasets = list(set(metrics['dataset_name']))
        details = {name: {k: [] for k in ['identifiers', 'epi_errs', 'R_errs', 't_errs']} for name in datasets}
        for k, i, e, r, t in zip(metrics['dataset_name'], metrics['identifiers'], metrics['epi_errs'], metrics['R_errs'], metrics['t_errs']):
            details[k]['identifiers'].append(i)
            details[k]['epi_errs'].append(e)
            details[k]['R_errs'].append(r)
            details[k]['t_errs'].append(t)

        details = {k: aggregate_metrics(v, self.tcfg.TRAINER.EPI_ERR_THR, test=test) for k, v in details.items() if len(v['epi_errs']) > 0}
        overall = aggregate_metrics(metrics, self.tcfg.TRAINER.EPI_ERR_THR, test=test)

        return overall, details

    def validation_epoch_end(self, outputs) -> None:
        all([len(set(o['Metrics']['dataset_name'])) == 1 for o in outputs])
        dataset_dicts = {o['Metrics']['dataset_name'][0]: [p['Dicts'] for p in outputs if o['Metrics']['dataset_name'][0] == p['Metrics']['dataset_name'][0]] for o in outputs}
        dataset_dicts = {name:{k: [_me[k].detach().cpu() for _me in dicts] for k in dicts[0] if torch.is_tensor(dicts[0][k])} for name, dicts in dataset_dicts.items()}
        dataset_dicts = {name:{k: sum(v) / (len(v) + 1e-8) for k, v in dicts.items()} for name, dicts in dataset_dicts.items()}

        overall, details = self.aggregate_metrics(outputs)

        self.log_dict({'Valid_Loss': dataset_dicts['MegaDepth']['Valid Total Loss']})
        self.log_dict({'Prec': details['MegaDepth']['Prec@5e-04']})
        self.log_dict({'AUC': details['MegaDepth']['AUC@5']})

        if self.trainer.global_rank == 0:
            for k0, v0 in details.items():  # k0: 'MegaDepth', 'ScanNet', ...
                for k1, v1 in v0.items():  # k1: 'AUC@5', 'AUC@10', 'AUC@20', 'Prec@5e-04'
                    title = "{}/{}".format(k0, k1)
                    self.logger.log_metrics({title: v1}, self.valid_step)
            for name, dicts in dataset_dicts.items():
                dicts = {name + '/' + k: v for k, v in dicts.items()}
                self.logger.log_metrics(dicts, self.valid_step)
            self.logger.log_metrics({'Valid step': self.valid_step}, self.valid_step)
            self.valid_step += 1

        self.log_valids = {k: 0 for k in self.log_valids.keys()}
