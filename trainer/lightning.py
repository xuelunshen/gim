# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import math
import torch

import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from modules.superpoint import SuperPoint
from modules.lightglue import LightGlue
from modules.utils.supervision import spvs
from tools.comm import all_gather
from tools.metrics import aggregate_metrics
from tools.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors
from tools.misc import lower_config, flattenList
from trainer.optimizer import build_optimizer, get_lr_scheduler


class Trainer(pl.LightningModule):

    def __init__(self, pcfg, tcfg, dcfg, ncfg):
        super().__init__()

        self.save_hyperparameters()
        self.pcfg = pcfg
        self.tcfg = tcfg
        self.ncfg = ncfg
        ncfg = lower_config(ncfg)
        self.superpoint = SuperPoint({
            'max_num_keypoints': 2048,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        })
        self.model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })

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
        optimizer = build_optimizer(self.model, self.tcfg)
        scheduler = get_lr_scheduler(optimizer)

        dic = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

        return dic

    def learning_rate_step(self):
        min_lr = self.pcfg.min_lr
        TRAINER = self.tcfg.TRAINER
        optimizer = self.trainer.optimizers[0]
        one_epoch_step = math.ceil(len(self.trainer.datamodule.trainer.train_dataloader) / self.pcfg.accumulate_grad_batches)
        all_epoch_step = one_epoch_step * self.pcfg.max_epochs
        # warmup_step = one_epoch_step
        warmup_step = self.pcfg.warmup_steps
        if self.trainer.global_step < warmup_step:
            base_lr = TRAINER.WARMUP_RATIO * TRAINER.TRUE_LR
            lr = base_lr + \
                 (self.trainer.global_step / warmup_step) * \
                 abs(TRAINER.TRUE_LR - base_lr)
        else:
            base_lr = all_epoch_step / (all_epoch_step - warmup_step) * (TRAINER.TRUE_LR - min_lr) + min_lr
            remain_step = all_epoch_step - self.trainer.global_step
            lr = remain_step / all_epoch_step * (base_lr - min_lr) + min_lr

        # p = 0.05
        p = 0
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
        pred = {}
        pred.update({k+'0': v for k, v in self.superpoint({
            "gt": data["gt"],
            "zs": data["zs"],
            "image": data["image0"],
            "image_size": data["resize0"][:, [1, 0]],
            "pseudo_labels": data["pseudo_labels"][..., :2]
        }).items()})
        pred.update({k+'1': v for k, v in self.superpoint({
            "gt": data["gt"],
            "zs": data["zs"],
            "image": data["image1"],
            "image_size": data["resize1"][:, [1, 0]],
            "pseudo_labels": data["pseudo_labels"][..., 2:]
        }).items()})
        pred.update(self.model({**pred, **data}))

        if data['gt'].sum():
            gt_pred = spvs({**pred, **data})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

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
            dicts = {'Learning rate': self.optimizers().param_groups[0]['lr'],
                     'step': self.train_step}
            dicts = {**dicts, **details}
            dicts = {'Train/' + k: v.detach().mean() if torch.is_tensor(v) else v for k, v in dicts.items()}
            self.logger.log_metrics(dicts, self.train_step)
            self.train_step += 1

    def training_step(self, batch, batch_idx):
        # self.learning_rate_step()
        pred = self.forward(batch)
        details, loss = self.loss(pred, batch)
        self.train_log(batch_idx, batch, details)
        return loss

    def loss(self, pred, data):
        loss = 0
        details = {}

        gt = data['gt']
        Ng = gt.sum().item()
        if Ng:
            pred_gt = {k: v[gt] if v.size(0) == gt.size(0) else v for k, v in pred.items() if isinstance(v, torch.Tensor)}
            data_gt = {k: v[gt] if v.size(0) == gt.size(0) else v for k, v in data.items() if isinstance(v, torch.Tensor)}
            losses, _ = self.model.loss(pred_gt, {**pred_gt, **data_gt})
            details.update({k:v.clone().detach().mean() for k, v in losses.items() if isinstance(v, torch.Tensor)})
            groudtruth_loss = torch.mean(losses["total"]) * (Ng / gt.size(0))
            details.update({'Groundtruth loss': groudtruth_loss.clone().detach()})
            loss += groudtruth_loss

        zs = data['zs']
        Nz = zs.sum().item()
        if Nz:
            scores = pred['log_assignment'][zs]
            pseudo_labels = data['pseudo_labels'][zs]
            b_ids, n_ids = torch.where(pseudo_labels.sum(dim=2) > 0)
            num_list = [torch.sum(b_ids == i).item() for i in range(Nz)]

            scores = [s[:m, :m] for s, m in zip(scores, num_list)]
            scores = [(-x.diagonal()).mean() for x in scores]
            pseudo_loss = torch.stack(scores).mean() * (Nz / zs.size(0))
            details.update({'Pseudo loss': pseudo_loss.clone().detach()})
            loss += pseudo_loss

        return details, loss

    def training_epoch_end(self, outputs) -> None:
        os.environ['EPOCH'] = str(self.current_epoch + 1)

    def valid_log(self, batch_idx, data, details):
        dicts = dict()
        dicts = {**dicts, **details}
        dicts = {'Valid ' + k: v.detach().mean() if torch.is_tensor(v) else v for k, v in dicts.items()}

        return dicts

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        pred.update(self.inference({**pred, **batch}))
        details, loss = self.loss(pred, batch)
        metrics = self.compute_metrics({**pred, **batch})
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

        self.log_dict({'Valid_Loss': dataset_dicts['MegaDepth']['Valid total']})
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

    def inference(self, data):
        # self.model.inference(data)
        mkpts0_f = torch.cat([kp * s for kp, s in zip(data['keypoints0'], data['scale0'][:, None])])
        mkpts1_f = torch.cat([kp * s for kp, s in zip(data['keypoints1'], data['scale1'][:, None])])
        m_bids = torch.nonzero(data['keypoints0'].sum(dim=2) > -1)[:, 0]
        matches = data['matches'][0]
        mkpts0_f = mkpts0_f[matches[..., 0]]  # coordinates in image #0, shape (K,2)
        mkpts1_f = mkpts1_f[matches[..., 1]]  # coordinates in image #1, shape (K,2)
        m_bids = m_bids[matches[..., 0]]
        return {
            'm_bids': m_bids,
            'mkpts0_f': mkpts0_f,
            'mkpts1_f': mkpts1_f,
        }
