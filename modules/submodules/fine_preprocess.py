import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        radius = W // 2
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})

        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        feat_fw = []

        # 2. select only the predicted matches
        feat_f0_g, feat_f1_g = None, None
        if data['gt'].sum():
            gt = data['gt']
            mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']
            i_ids = mkpts0_c[:, 0] + mkpts0_c[:, 1] * data['hw0_c'][1]
            j_ids = mkpts1_c[:, 0] + mkpts1_c[:, 1] * data['hw1_c'][1]
            feat_f0_g = feat_f0_unfold[gt][data['b_ids'], i_ids]  # [n, ww, cf]
            feat_f1_g = feat_f1_unfold[gt][data['b_ids'], j_ids]

        feat_f0_z, feat_f1_z = None, None
        if data['zs'].sum():
            zs = data['zs']
            pt0_i = data['zs_pt0_i']
            pt1_i = data['zs_pt1_i']
            zs_b_ids = data['zs_b_ids']
            scale_c = data['hw0_i'][0] / data['hw0_c'][0]  # 8.0
            scale_f = data['hw0_i'][0] / data['hw0_f'][0]  # 2.0
            pt0_c_int = (pt0_i / scale_c).round().long()
            pt1_c_int = (pt1_i / scale_c).round().long()
            pt0_f_int = pt0_c_int * stride
            pt1_f_int = pt1_c_int * stride
            pt0_f_float = pt0_i / scale_f
            pt1_f_float = pt1_i / scale_f
            indices = ((pt0_f_float[:, 0] - pt0_f_int[:, 0]).abs() <= radius) & \
                      ((pt0_f_float[:, 1] - pt0_f_int[:, 1]).abs() <= radius) & \
                      ((pt1_f_float[:, 0] - pt1_f_int[:, 0]).abs() <= radius) & \
                      ((pt1_f_float[:, 1] - pt1_f_int[:, 1]).abs() <= radius)
            zs_ci_ids = pt0_c_int[:, 0] + pt0_c_int[:, 1] * data['hw0_c'][1]
            zs_cj_ids = pt1_c_int[:, 0] + pt1_c_int[:, 1] * data['hw1_c'][1]
            feat_f0_z = feat_f0_unfold[zs][zs_b_ids[indices], zs_ci_ids[indices]]  # [n, ww, cf]
            feat_f1_z = feat_f1_unfold[zs][zs_b_ids[indices], zs_cj_ids[indices]]

            data.update({
                'zs_b_ids': zs_b_ids[indices],
                'zs_pt0_f_int': pt0_f_int[indices],
                'zs_pt1_f_int': pt1_f_int[indices],
                'zs_pt0_f_float': pt0_f_float[indices],
                'zs_pt1_f_float': pt1_f_float[indices],
            })

        if data['gt'].sum() and data['zs'].sum():
            feat_fw.append(feat_f0_g)
            feat_fw.append(feat_f0_z)
            feat_fw.append(feat_f1_g)
            feat_fw.append(feat_f1_z)
        elif data['gt'].sum():
            feat_fw.append(feat_f0_g)
            feat_fw.append(feat_f1_g)
        elif data['zs'].sum():
            feat_fw.append(feat_f0_z)
            feat_fw.append(feat_f1_z)

        feat_fw = torch.cat(feat_fw, dim=0)  # [n+m+n+m, ww, cf]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:

            feat_cw = []

            feat_c0_g, feat_c1_g = None, None
            if data['gt'].sum():
                gt = data['gt']
                feat_c0_g = feat_c0[gt][data['b_ids'], data['i_ids']]  # (n, 256)
                feat_c1_g = feat_c1[gt][data['b_ids'], data['j_ids']]

            if data['gt'].sum() and data['zs'].sum():
                feat_cw.append(feat_c0_g)
                feat_cw.append(data['zs_feat_c0'])
                feat_cw.append(feat_c1_g)
                feat_cw.append(data['zs_feat_c1'])
            elif data['gt'].sum():
                feat_cw.append(feat_c0_g)
                feat_cw.append(feat_c1_g)
            elif data['zs'].sum():
                feat_cw.append(data['zs_feat_c0'])
                feat_cw.append(data['zs_feat_c1'])

            feat_c_win = torch.cat(feat_cw, dim=0)  # (n+m+n+m, 256)
            feat_c_win = self.down_proj(feat_c_win)  # [n+m+n+m, 128]

            feat_cf_win = torch.cat([
                feat_fw,  # [n+m+n+m, ww, 128]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [n+m+n+m, ww, 128]
            ], -1)  # [n+m+n+m, ww, 256]
            feat_cf_win = self.merge_feat(feat_cf_win)  # [n+m+n+m, ww, 128]
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)  # (n+m, ww, cf), (n+m, ww, cf)
        else:
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_fw, 2, dim=0)  # (n+m, ww, cf), (n+m, ww, cf)

        return feat_f0_unfold, feat_f1_unfold
