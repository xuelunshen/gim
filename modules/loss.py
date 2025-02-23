from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops.einops import rearrange

from datasets.walk.walk import pt_to_grid
from modules.utils.supervision import Flip
from modules.utils.geometry import warp_kpts


class DepthRegressionLoss(nn.Module):
    def __init__(
        self,
        robust=True,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
    ):
        super().__init__()
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale

    def geometric_dist(self, data, dense_matches):
        with torch.no_grad():
            gt = data['gt']
            device = dense_matches.device
            N, h0, w0, d = dense_matches.shape
            H0, W0 = data['image0'].shape[-2:]
            scale0 = data['scale0'][:, None][gt]
            scale1 = data['scale1'][:, None][gt]

            Hq_aug = data['Hq_aug'][gt]  # (bs, 3, 3)
            offset0, offset1 = data['offset0'][:, None][gt], data['offset1'][:, None][gt]  # (bs, 1, 2) - <x, y>

            grid_pt0 = torch.meshgrid(*[(torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) + 1) / 2 * m for n, m in zip([h0, w0], [H0, W0])])
            grid_pt0_fs = torch.stack((grid_pt0[1], grid_pt0[0]), dim=-1)[None].expand(N, h0, w0, 2).reshape(N, h0 * w0, 2)
            grid_pt0_rs = Flip(grid_pt0_fs, data['hflip0'][gt], data['vflip0'][gt], data['resize0'][:, 1][gt] - 1, data['resize0'][:, 0][gt] - 1)  # rectified flip
            grid_pt0_rs += offset0
            grid_pt0_i = grid_pt0_rs * scale0

            pos_mask0, neg_mask0, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'][gt], data['depth1'][gt], data['T_0to1'][gt], data['K0'][gt], data['K1'][gt], data['K0_'][gt], data['K1_'][gt], data['imsize1'][gt])

            w_pt0_s = w_pt0_i / scale1
            w_pt0_s = w_pt0_s - offset1
            w_pt0_s = Flip(w_pt0_s, data['hflip1'][gt], data['vflip1'][gt], data['resize1'][:, 1][gt] - 1, data['resize1'][:, 0][gt] - 1)
            w_pt0_s = torch.cat([w_pt0_s, torch.ones_like(w_pt0_s[:, :, :1])], dim=-1)
            w_pt0_s = torch.einsum('bij,bjk->bik', w_pt0_s, Hq_aug.transpose(1, 2))
            w_pt0_s = w_pt0_s[..., :2] / (w_pt0_s[..., [2]] + 1e-8)
            w_pt0_s[..., 0] = (w_pt0_s[..., 0] / (W0 - 1)) * 2 - 1
            w_pt0_s[..., 1] = (w_pt0_s[..., 1] / (H0 - 1)) * 2 - 1

            prob = pos_mask0.float().reshape(N, h0, w0)

        gd = (dense_matches - w_pt0_s.reshape(N, h0, w0, 2)).norm(dim=-1)  # *scale?

        return gd, prob

    def zeroshot_dist(self, data, dense_matches, b_ids, grid0, grid1, znum):
        """

        Args:
            data:
            dense_matches: (N, 2, h0, w0)
            b_ids: (n,)
            grid0: (1, 1, n, 2) in [-1, 1]
            grid1: (n, 2) in [-1, 1]
            znum: int

        Returns:

        """
        grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')

        pred = [grid_sample(dense_matches[[i]], grid0[:, :, b_ids == i]) for i in range(znum)]  # [(1, 2, 1, n)]
        pred = torch.cat([x.squeeze().transpose(0, 1) for x in pred], dim=0)  # (n, 2) in [-1, 1]

        gd = (pred - grid1).norm(dim=-1)  # (n,)

        gd[torch.isnan(gd)] = 0
        gd[torch.isinf(gd)] = 0

        return gd

    def zeroshot_ce(self, dense_certainty, b_ids, grid0, znum):
        """

        Args:
            dense_certainty: (N, 1, h0, w0)
            b_ids: (n,)
            grid0: (1, 1, n, 2) in [-1, 1]
            znum: int

        Returns:

        """
        grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')

        pred = [grid_sample(dense_certainty[[i]], grid0[:, :, b_ids == i]) for i in range(znum)]  # [(1, 1, 1, n)]
        pred = torch.cat([x.squeeze() for x in pred], dim=0)  # (n) in [0, 1]

        ce_loss = F.binary_cross_entropy_with_logits(pred, pred.new_ones(pred.size()), reduction='none')
        ce_loss[torch.isnan(ce_loss)] = 0
        ce_loss[torch.isinf(ce_loss)] = 0
        ce_loss = torch.stack([ce_loss[b_ids == i].mean() for i in range(znum)])

        return ce_loss

    def dense_depth_loss(self, dense_certainty, prob, gd, scale, eps=1e-8):
        smooth_prob = prob
        ce_loss = F.binary_cross_entropy_with_logits(dense_certainty[:, 0], smooth_prob, reduction='none')
        depth_loss = gd[prob > 0]
        if not torch.any(prob > 0).item():
            depth_loss = gd * 0.0  # Prevent issues where prob is 0 everywhere
        ce_loss[torch.isnan(ce_loss)] = 0
        ce_loss[torch.isinf(ce_loss)] = 0
        ce_loss = ce_loss.mean(dim=(1, 2))
        depth_loss[torch.isnan(depth_loss)] = 0
        depth_loss[torch.isinf(depth_loss)] = 0
        return {
            f"CE Loss [{scale}]": ce_loss,
            f"Depth Loss [{scale}]": depth_loss.mean(),
        }

    def forward(self, batch):
        """[summary]

        Args:
            batch ([dict: 9]):
                "dense_corresps" ([dict: 6]): has keys: 1, 2, 4, 8, 16, 32
                    each item has a {dict: 2} with keys
                    "dense_certainty": [b, 1, h/scale, w/scale], (h, w) is the network input size of image
                    "dense_flow": [b, 2, h/scale, w/scale]
                "K1": [b, 3, 3]
                "K2": [b, 3, 3]
                "query": [b, 3, h, w]
                "query_depth": [b, h, w]
                "query_identifier" (list): length is b
                "support": [b, 3, h, w]
                "support_depth": [b, h, w]
                "support_identifier" (list): length is b
                "T_1to2": [b, 4, 4]

        Returns:
            [type]: [description]
        """

        loss = []
        loss_scalars = {}

        dense_corresps = batch["dense_corresps"]
        scales = list(dense_corresps.keys())

        gt = batch['gt']
        if gt.sum() > 0:
            gt_loss = 0.0
            prev_gd = None
            not_scannet_mask = gt.new_tensor([x != 'ScanNet' for x in batch['dataset_name']])[gt]
            for scale in scales:
                dense_scale_certainty, dense_scale_coords = \
                    dense_corresps[scale]["dense_certainty"][gt], \
                    dense_corresps[scale]["dense_flow"][gt]
                dense_scale_coords = rearrange(dense_scale_coords,
                                               "b d h w -> b h w d")  # [b, h/scale, w/scale, 2]
                h, w = dense_scale_coords.shape[1:3]
                gd, prob = self.geometric_dist(batch, dense_scale_coords)
                if scale <= self.local_largest_scale and self.local_loss:  # scale <= 8 and True # Thought here is that fine matching loss should not be punished by coarse mistakes, but should identify wrong matching
                    prob = prob * (
                            F.interpolate(prev_gd[:, None], size=(h, w), mode="nearest")[:, 0]
                            < (2 / 512) * (self.local_dist * scale)
                    )
                depth_losses = self.dense_depth_loss(dense_scale_certainty, prob, gd, scale)
                ce_loss = depth_losses[f"CE Loss [{scale}]"]
                # if torch.isnan(ce_loss).any():
                #     ce_loss = 0
                dp_loss = depth_losses[f"Depth Loss [{scale}]"]
                # if torch.isnan(dp_loss).any():
                #     dp_loss = 0
                scale_loss = (self.ce_weight * not_scannet_mask * ce_loss).sum() / (not_scannet_mask.sum() + 1e-8) + dp_loss# scale ce loss for coarser scales
                if self.scale_normalize:  # self.scale_normalize is False
                    scale_loss = scale_loss * 1 / scale
                gt_loss = gt_loss + scale_loss
                loss_scalars.update({'GT '+k: v.clone().detach()
                                     for k, v in depth_losses.items()})
                prev_gd = gd.detach()
            loss.append(gt_loss)

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
                dense_scale_certainty, dense_scale_coords = \
                    dense_corresps[scale]["dense_certainty"][zs], \
                    dense_corresps[scale]["dense_flow"][zs]
                gd = self.zeroshot_dist(batch, dense_scale_coords, b_ids, grid0, grid1, znum)
                dp_loss = gd.mean()
                # if torch.isnan(dp_loss).any():
                #     dp_loss = 0
                zs_loss = zs_loss + dp_loss
                ce = self.zeroshot_ce(dense_scale_certainty, b_ids, grid0, znum)
                ce_loss = (self.ce_weight * ce).sum() / (znum + 1e-8)
                # zs_loss = zs_loss + ce_loss
                loss_scalars.update({f"ZS Depth Loss [{scale}]": dp_loss.clone().detach()})
                loss_scalars.update({f"ZS CE Loss [{scale}]": ce_loss.clone().detach()})
            loss.append(zs_loss)

        loss = sum(loss) / len(loss)
        loss_scalars.update({'Total Loss': loss.clone().detach()})
        details = dict(
            loss=loss, loss_scalars=loss_scalars
        )
        return details
