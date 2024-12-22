import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops.einops import rearrange
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat_f0 (torch.Tensor): [M, WW, C]
            feat_f1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        W = data['W']
        radius = W // 2
        M, WW, C = feat_f0.shape

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training is False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        Ng = len(data['b_ids']) if 'b_ids' in data else 0
        Nz = len(data['zs_b_ids']) if 'zs_b_ids' in data else 0
        assert M == (Ng + Nz)

        heatmaps = []
        softmax_temp = 1. / C ** .5

        if data['gt'].sum():
            feat_f0_picked = feat_f0[:Ng][:, WW // 2, :]  # (Ng, c)
            sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1[:Ng])  # (Ng, ww)
            heatmap_g = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)  # (Ng, w, w)
            heatmaps.append(heatmap_g)

        if data['zs'].sum():
            # scale = data['hw0_i'][0] / data['hw0_f'][0]  # 2
            # pt0_f_float = data['zs_pt0_i'] / scale  # (Nz, 2) in hw_f coordinates
            pt0_f_int = data['zs_pt0_f_int']
            pt0_f_float = data['zs_pt0_f_float']  # (Nz, 2) in hw_f coordinates
            # pt_x = (pt0_f_float[:, 0] - pt0_f_int[:, 0] + radius) / (W - 1) * 2 - 1
            # pt_y = (pt0_f_float[:, 1] - pt0_f_int[:, 1] + radius) / (W - 1) * 2 - 1
            pt_x = (pt0_f_float[:, 0] - pt0_f_int[:, 0]) / radius
            pt_y = (pt0_f_float[:, 1] - pt0_f_int[:, 1]) / radius
            grid = torch.stack([pt_x, pt_y], dim=1)[:, None, None]  # (Nz, 1, 1, 2)
            grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')
            feat_f0_picked = rearrange(feat_f0[-Nz:], 'n (h w) c -> n c h w', h=W, w=W)
            feat_f0_picked = grid_sample(feat_f0_picked, grid).squeeze()  # [(Nz, c)]
            sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1[-Nz:])  # (Nz, ww)
            heatmap_z = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)  # (Nz, w, w)
            heatmaps.append(heatmap_z)

            self.spvc_zeroshot_fine(radius, data)

        heatmap = torch.cat(heatmaps, dim=0)  # (M, w, w)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        if data['gt'].sum(): self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W = data['W']
        gt = data['gt']
        b_ids = data['b_ids']
        Ng = len(data['b_ids'])
        scale = data['hw0_i'][0] / data['hw0_c'][0]  # 8
        scale0 = scale * data['scale0'][gt][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][gt][b_ids] if 'scale1' in data else scale

        # mkpts0_f and mkpts1_f
        scale = data['hw0_i'][0] / data['hw0_f'][0]  # 2
        scale = scale * data['scale1'][gt][b_ids] if 'scale1' in data else scale
        mkpts0_f = data['mkpts0_c'] * scale0
        mkpts1_f = data['mkpts1_c'] * scale1 + (coords_normed[:Ng] * (W // 2) * scale)

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })

    @torch.no_grad()
    def spvc_zeroshot_fine(self, radius, data):
        # scale = data['hw0_i'][0] / data['hw0_f'][0]  # 2
        # pt1_f_float = data['zs_pt1_i'] / scale
        pt1_f_int = data['zs_pt1_f_int']
        pt1_f_float = data['zs_pt1_f_float']
        expec_f_zs = (pt1_f_float - pt1_f_int) / radius
        data.update({"expec_f_zs": expec_f_zs})
