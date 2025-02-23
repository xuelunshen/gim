from loguru import logger

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops.einops import rearrange

from datasets.walk.walk import pt_to_grid
from modules.utils.supervision import Flip
from modules.utils.geometry import warp_kpts


class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask=False,
        depth_interpolation_mode="bilinear",
        mask_depth_loss=False,
        relative_depth_error_threshold=0.05,
        alpha=1.0,
        c=1e-3,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2)
            GT = (G[None, :, None, None, :] - x2[:, None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction="none")[prob > 0.99]

        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:, 0], prob)

        if not torch.any(cls_loss):
            # Prevent issues where prob is 0 everywhere
            cls_loss = certainty_loss * 0.0

        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def delta_cls_loss(
        self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale
    ):
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(
                *[
                    torch.linspace(
                        -1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device
                    )
                    for _ in range(2)
                ]
            )
            G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2) * offset_scale
            GT = (
                (G[None, :, None, None, :] + flow_pre_delta[:, None] - x2[:, None])
                .norm(dim=-1)
                .min(dim=1)
                .indices
            )
        cls_loss = F.cross_entropy(delta_cls, GT, reduction="none")[prob > 0.99]
        if not torch.any(cls_loss):
            # Prevent issues where prob is 0 everywhere
            # cls_loss = (certainty_loss * 0.0)
            pass  # TODO
        certainty_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode="delta"):
        epe = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1)
        # if scale == 1:
        #     pck_05 = (epe[prob > 0.99] < 0.5 * (2 / 512)).float().mean()

        certainty = certainty[:, 0]

        mask = (~torch.isnan(certainty)) & (~torch.isinf(certainty))
        if mask.sum():
            ce_loss = F.binary_cross_entropy_with_logits(certainty[mask], prob[mask]).mean()
        else:
            ce_loss = 0

        a = self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x / cs) ** 2 + 1**2) ** (a / 2)

        mask = (~torch.isnan(reg_loss)) & (~torch.isinf(reg_loss))
        if mask.sum():
            reg_loss = reg_loss[mask].mean()
        else:
            reg_loss = 0

        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss,
            f"{mode}_regression_loss_{scale}": reg_loss,
        }
        return losses

    def gim_regression_loss(self, b_ids, znum, grid0, grid1, flow, scale):
        """

        Args:
            b_ids: (n,)
            znum: int
            grid0: (1, 1, n, 2) in [-1, 1]
            grid1: (n, 2) in [-1, 1]
            flow: (N, 2, h0, w0)
            scale: float

        Returns:

        """
        grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')

        pred = [grid_sample(flow[[i]], grid0[:, :, b_ids == i]) for i in range(znum)]  # [(1, 2, 1, n)]
        pred = torch.cat([x.squeeze().transpose(0, 1) for x in pred], dim=0)  # (n, 2) in [-1, 1]

        epe = (pred - grid1).norm(dim=-1)  # (n,)

        # get mask that gd is not nan and inf
        mask = (~torch.isnan(epe)) & (~torch.isinf(epe))

        x = epe[mask]
        a = self.alpha
        cs = self.c * scale
        reg_loss = cs ** a * ((x / cs) ** 2 + 1 ** 2) ** (a / 2)

        mask = (~torch.isnan(reg_loss)) & (~torch.isinf(reg_loss))
        if mask.sum():
            reg_loss = reg_loss[mask].mean()
        else:
            reg_loss = 0

        return reg_loss

    def gim_cls_loss(self, b_ids, znum, grid0, grid1, scale_gm_cls, scale):
        grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')

        pred = [grid_sample(scale_gm_cls[[i]], grid0[:, :, b_ids == i]) for i in range(znum)]  # [(1, 4096, 1, n)]
        pred = torch.cat([x.squeeze().transpose(0, 1) for x in pred], dim=0)  # (n, 4096) in [-1, 1]

        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape  # (2, 4096, 40, 40)
            device = grid1.device
            cls_res = round(math.sqrt(C))  # 64
            G = torch.meshgrid(*[torch.linspace(-1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2)  # (4096, 2)
            GT = (grid1[:, None, :] - G[None]).norm(dim=-1).min(dim=1).indices  # (n,)

        cls_loss = F.cross_entropy(pred, GT, reduction="none")

        return cls_loss.mean()

    def forward(self, batch):
        loss = []
        loss_scalars = {}

        dense_corresps = batch['corresps']
        scales = list(dense_corresps.keys())

        gt = batch['gt']
        if gt.sum() > 0:
            gt_loss = 0.0
            prev_epe = None
            # scale_weights due to differences in scale for regression gradients and classification gradients
            scale_weights = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
            for scale in scales:
                scale_corresps = dense_corresps[scale]

                flow_pre_delta = scale_corresps["flow_pre_delta"][gt]  # (2, 2, 40, 40)
                device = flow_pre_delta.device
                b, d, h, w = flow_pre_delta.shape  # 2, 2, 40, 40
                gt_warp, gt_prob = get_gt_warp(batch, b, h, w, device)  # (2, 40, 40, 2), (2, 40, 40)
                x2 = gt_warp.float()  # (2, 40, 40, 2)
                prob = gt_prob  # (2, 40, 40)

                if self.local_largest_scale >= scale and prev_epe is not None:  # self.local_largest_scale is 8
                    prob = prob * (
                        F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[
                            :, 0
                        ]
                        < (2 / 512) * (self.local_dist[scale] * scale)
                    )

                if 'gm_cls' in scale_corresps:
                    scale_gm_cls = scale_corresps.get("gm_cls")[gt]  # (2, 4096, 40, 40)
                    scale_gm_certainty = scale_corresps.get("gm_certainty")[gt]  # (2, 1, 40, 40)
                    gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                    gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                    gt_loss = gt_loss + scale_weights[scale] * gm_loss
                    loss_scalars.update({f"gt_gm_loss_{scale}": gm_loss.clone().detach()})

                flow = scale_corresps["flow"][gt]  # (2, 2, 40, 40)
                scale_certainty = scale_corresps["certainty"][gt]  # (2, 1, 40, 40)
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                gt_loss = gt_loss + scale_weights[scale] * reg_loss
                if isinstance(reg_loss, torch.Tensor):
                    loss_scalars.update({f"gt_reg_loss_{scale}": reg_loss.clone().detach()})
                prev_epe = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1).detach()
            loss.append(gt_loss)
            loss_scalars.update({'gt total loss': gt_loss.clone().detach()})

        zs = batch['zs']
        if zs.sum() > 0:
            znum = zs.sum()
            pseudo_labels = batch['pseudo_labels'][zs]
            b_ids, n_ids = torch.where(pseudo_labels.sum(dim=2) > 0)
            pseudo_labels = pseudo_labels[b_ids, n_ids]  # (n, 4)
            pt0 = pseudo_labels[:, :2]  # (n, 2), in hw_i(image) size coordinates
            pt1 = pseudo_labels[:, 2:]  # (n, 2), in hw_i(image) size coordinates

            sample_num = 2048
            unique_b = torch.unique(b_ids)
            if (sample_num > 0) and len(b_ids) > (sample_num * len(unique_b)):
                indices = torch.cat([
                    torch.randperm((b_ids == b).sum(), device=pseudo_labels.device)[:sample_num]
                    + (b_ids < b).sum()
                    for b in unique_b])
                b_ids, pt0, pt1 = b_ids[indices], pt0[indices], pt1[indices]

            grid0 = pt_to_grid(pt0.clone()[None], batch['hw0_i'])  # (1, 1, n, 2) in [-1, 1]
            grid1 = pt_to_grid(pt1.clone()[None], batch['hw1_i']).squeeze()  # (n, 2) in [-1, 1]

            zs_loss = 0.0
            for scale in scales:
                flow = dense_corresps[scale]["flow"][zs]  # (2, 2, 40, 40)
                scale_certainty = dense_corresps[scale]["certainty"][zs]
                reg_loss = self.gim_regression_loss(b_ids, znum, grid0, grid1, flow, scale)
                zs_loss = zs_loss + reg_loss
                if isinstance(reg_loss, torch.Tensor):
                    loss_scalars.update({f"zs_reg_loss_{scale}": reg_loss.clone().detach()})

                if 'gm_cls' in dense_corresps[scale]:
                    scale_gm_cls = dense_corresps[scale].get("gm_cls")[zs]  # (2, 4096, 40, 40)
                    gm_loss = self.gim_cls_loss(b_ids, znum, grid0, grid1, scale_gm_cls, scale)
                    zs_loss = zs_loss + gm_loss
                    loss_scalars.update({f"zs_gm_loss_{scale}": gm_loss.clone().detach()})

            loss.append(zs_loss)
            loss_scalars.update({'zs total loss': zs_loss.clone().detach()})

        loss = sum(loss) / len(loss)
        loss_scalars.update({'Total Loss': loss.clone().detach()})
        details = dict(loss=loss, loss_scalars=loss_scalars)

        return details
