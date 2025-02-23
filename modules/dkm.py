# -*- coding: utf-8 -*-
# @Author  : Parskatt

import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from einops import rearrange


def get_model(pretrained_backbone=True, h=None, w=None):
    gp_dim = 256
    dfn_dim = 384
    feat_dim = 256
    coordinate_decoder = DFN(
        internal_dim=dfn_dim,
        feat_input_modules=nn.ModuleDict(
            {
                "32": nn.Conv2d(512, feat_dim, 1, 1),
                "16": nn.Conv2d(512, feat_dim, 1, 1),
            }
        ),
        pred_input_modules=nn.ModuleDict(
            {
                "32": nn.Identity(),
                "16": nn.Identity(),
            }
        ),
        rrb_d_dict=nn.ModuleDict(
            {
                "32": RRB(gp_dim + feat_dim, dfn_dim),
                "16": RRB(gp_dim + feat_dim, dfn_dim),
            }
        ),
        cab_dict=nn.ModuleDict(
            {
                "32": CAB(2 * dfn_dim, dfn_dim),
                "16": CAB(2 * dfn_dim, dfn_dim),
            }
        ),
        rrb_u_dict=nn.ModuleDict(
            {
                "32": RRB(dfn_dim, dfn_dim),
                "16": RRB(dfn_dim, dfn_dim),
            }
        ),
        terminal_module=nn.ModuleDict(
            {
                "32": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
                "16": nn.Conv2d(dfn_dim, 3, 1, 1, 0),
            }
        ),
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    local_corr = True
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
            ),
            "1": ConvRefiner(
                2 * 3 + 6,
                24,
                3,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"32": gp32, "16": gp16})
    proj = nn.ModuleDict(
        {"16": nn.Conv2d(1024, 512, 1, 1), "32": nn.Conv2d(2048, 512, 1, 1)}
    )
    decoder = Decoder(coordinate_decoder, gps, proj, conv_refiner, detach=True)
    encoder = ResNet50(pretrained=pretrained_backbone, high_res=False, freeze_bn=False)
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, alpha=1, beta=0)
    return matcher


def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow = None
):
    device = feature0.device
    b, c, h, w = feature0.size()
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
                ))
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
    else:
        coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    r = local_radius
    local_window = torch.meshgrid(
                (
                    torch.linspace(-2*local_radius/h, 2*local_radius/h, 2*r+1, device=device),
                    torch.linspace(-2*local_radius/w, 2*local_radius/w, 2*r+1, device=device),
                ))
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(b, 2*r+1, 2*r+1, 2).reshape(b, (2*r+1)**2, 2)
    coords = (coords[:,:,:,None]+local_window[:,None,None]).reshape(b,h,w*(2*r+1)**2,2)
    window_feature = F.grid_sample(
        feature1, coords, padding_mode=padding_mode, align_corners=False
    )[...,None].reshape(b,c,h,w,(2*r+1)**2)
    corr = torch.einsum("bchw, bchwk -> bkhw", feature0, window_feature)/(c**.5)
    return corr


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently supported in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        if self.predict_features:
            f = f + y[:,:self.dim] # Stupid way to predict features
        b, d, h2, w2 = f.shape
        #assert x.shape == y.shape
        x, y, f = self.reshape(x), self.reshape(y), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        # Due to https://github.com/pytorch/pytorch/issues/16963 annoying warnings, remove batch if N large
        if len(K_yy[0]) > 2000:
            K_yy_inv = torch.cat([torch.linalg.inv(K_yy[k:k+1] + sigma_noise[k:k+1]) for k in range(b)])
        else:
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats


class DFN(nn.Module):
    def __init__(
        self,
        internal_dim,
        feat_input_modules,
        pred_input_modules,
        rrb_d_dict,
        cab_dict,
        rrb_u_dict,
        use_global_context=False,
        global_dim=None,
        terminal_module=None,
        upsample_mode="bilinear",
        align_corners=False,
    ):
        super().__init__()
        if use_global_context:
            assert (
                global_dim is not None
            ), "Global dim must be provided when using global context"
        self.align_corners = align_corners
        self.internal_dim = internal_dim
        self.feat_input_modules = feat_input_modules
        self.pred_input_modules = pred_input_modules
        self.rrb_d = rrb_d_dict
        self.cab = cab_dict
        self.rrb_u = rrb_u_dict
        self.use_global_context = use_global_context
        if use_global_context:
            self.global_to_internal = nn.Conv2d(global_dim, self.internal_dim, 1, 1, 0)
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.terminal_module = (
            terminal_module if terminal_module is not None else nn.Identity()
        )
        self.upsample_mode = upsample_mode
        self._scales = [int(key) for key in self.terminal_module.keys()]

    def scales(self):
        return self._scales.copy()

    def forward(self, embeddings, feats, context, key):
        feats = self.feat_input_modules[str(key)](feats)
        embeddings = torch.cat([feats, embeddings], dim=1)
        embeddings = self.rrb_d[str(key)](embeddings)
        context = self.cab[str(key)]([context, embeddings])
        context = self.rrb_u[str(key)](context)
        preds = self.terminal_module[str(key)](context)
        pred_coord = preds[:, -2:]
        pred_certainty = preds[:, :-2]
        return pred_coord, pred_certainty, context


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low (old, new)
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class Decoder(nn.Module):
    def __init__(
        self, embedding_decoder, gps, proj, conv_refiner, transformers = None, detach=False, scales="all", pos_embeddings = None,
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales

    def upsample_preds(self, flow, certainty, query, support):
        b, hs, ws, d = flow.shape
        b, c, h, w = query.shape
        flow = flow.permute(0, 3, 1, 2)
        certainty = F.interpolate(
            certainty, size=(h, w), align_corners=False, mode="bilinear"
        )
        flow = F.interpolate(
            flow, size=(h, w), align_corners=False, mode="bilinear"
        )
        delta_certainty, delta_flow = self.conv_refiner["1"](query, support, flow)
        flow = torch.stack(
                (
                    flow[:, 0] + delta_flow[:, 0] / (4 * w),
                    flow[:, 1] + delta_flow[:, 1] / (4 * h),
                ),
                dim=1,
            )
        flow = flow.permute(0, 2, 3, 1)
        certainty = certainty + delta_certainty
        return flow, certainty

    def get_placeholder_flow(self, b, h, w, device):
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            )
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def forward(self, f1, f2, upsample = False, dense_flow = None, dense_certainty = None):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"]
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b, self.embedding_decoder.internal_dim, *sizes[coarsest_scale], device=f1[coarsest_scale].device
        )
        dense_corresps = {}
        if not upsample:
            dense_flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            dense_certainty = 0.0
        else:
            dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
            dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[coarsest_scale],
                    align_corners=False,
                    mode="bilinear",
                )
        for new_scale in all_scales:
            ins = int(new_scale)
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)
            b, c, hs, ws = f1_s.shape
            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                new_stuff = self.gps[new_scale](f1_s, f2_s, dense_flow=dense_flow)
                dense_flow, dense_certainty, old_stuff = self.embedding_decoder(
                    new_stuff, f1_s, old_stuff, new_scale
                )

            if new_scale in self.conv_refiner:
                delta_certainty, displacement = self.conv_refiner[new_scale](
                    f1_s, f2_s, dense_flow
                )
                dense_flow = torch.stack(
                    (
                        dense_flow[:, 0] + ins * displacement[:, 0] / (4 * w),
                        dense_flow[:, 1] + ins * displacement[:, 1] / (4 * h),
                    ),
                    dim=1,
                )
                dense_certainty = (
                    dense_certainty + delta_certainty
                )  # predict both certainty and displacement

            dense_corresps[ins] = {
                "dense_flow": dense_flow,
                "dense_certainty": dense_certainty,
            }

            if new_scale != "1":
                dense_flow = F.interpolate(
                    dense_flow,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )

                dense_certainty = F.interpolate(
                    dense_certainty,
                    size=sizes[ins // 2],
                    align_corners=False,
                    mode="bilinear",
                )
                if self.detach:
                    dense_flow = dense_flow.detach()
                    dense_certainty = dense_certainty.detach()
        return dense_corresps


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res=False, weights=None, dilation=None,
                 freeze_bn=True, anti_aliased=False) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights=weights,
                                        replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,
                                        replace_stride_with_dilation=dilation)

        del self.net.fc
        self.high_res = high_res
        self.freeze_bn = freeze_bn

    def forward(self, x):
        net = self.net
        feats = {1: x}
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x
        x = net.layer2(x)
        feats[8] = x
        x = net.layer3(x)
        feats[16] = x
        x = net.layer4(x)
        feats[32] = x
        return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass


class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_radius = None,
        corr_in_other = None,
        no_support_fm = False,
    ):
        super().__init__()
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_support_fm = no_support_fm

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
        )
        norm = nn.BatchNorm2d(out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow):
        """Computes the relative refining displacement in pixels for a given image x,y and a coarse flow-field between them

        Args:
            x ([type]): [description]
            y ([type]): [description]
            flow ([type]): [description]

        Returns:
            [type]: [description]
        """
        device = x.device
        b,c,hs,ws = x.shape
        with torch.no_grad():
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False)
        if self.has_displacement_emb:
            query_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                )
            )
            query_coords = torch.stack((query_coords[1], query_coords[0]))
            query_coords = query_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow-query_coords
            emb_in_displacement = self.disp_emb(in_displacement)
            if self.local_corr_radius:
                #TODO: should corr have gradient?
                if self.corr_in_other:
                    # Corr in other means take a kxk grid around the predicted coordinate in other image
                    local_corr = local_correlation(x,y,local_radius=self.local_corr_radius,flow = flow)
                else:
                    # Otherwise we use the warp to sample in the first image
                    # This is actually different operations, especially for large viewpoint changes
                    local_corr = local_correlation(x, x_hat, local_radius=self.local_corr_radius,)
                if self.no_support_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:
            if self.no_support_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d)
        certainty, displacement = d[:, :-2], d[:, -2:]
        return certainty, displacement


class RegressionMatcher(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            h=384,
            w=512,
            use_contrastive_loss=False,
            alpha=1,
            beta=0,
            sample_mode="threshold",
            upsample_preds=False,
            symmetric=False,
            name=None,
            use_soft_mutual_nearest_neighbours=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w_resized = w
        self.h_resized = h
        self.use_contrastive_loss = use_contrastive_loss
        self.alpha = alpha
        self.beta = beta
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.symmetric = symmetric
        self.name = name
        self.sample_thresh = 0.05
        self.upsample_res = (864, 1152)
        if use_soft_mutual_nearest_neighbours:
            assert symmetric, "MNS requires symmetric inference"
        self.use_soft_mutual_nearest_neighbours = use_soft_mutual_nearest_neighbours

    def extract_backbone_features(self, batch, batched=True, upsample=True):
        # TODO: only extract stride [1,2,4,8] for upsample = True
        x_q = batch["color0"]
        x_s = batch["color1"]
        if batched:
            X = torch.cat((x_q, x_s))
            feature_pyramid = self.encoder(X)
        else:
            feature_pyramid = self.encoder(x_q), self.encoder(x_s)
        return feature_pyramid

    def sample(
            self,
            dense_matches,
            dense_certainty,
            num=10000,
    ):
        b = dense_certainty.size(0)
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh  # Hit this
            dense_certainty = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        elif "pow" in self.sample_mode:
            dense_certainty = dense_certainty ** (1 / 3)
        elif "naive" in self.sample_mode:
            dense_certainty = torch.ones_like(dense_certainty)
        matches, certainty = (
            dense_matches.reshape(b, -1, 4),
            dense_certainty.reshape(b, -1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1  # expansion_factor = 1
        if not all(certainty.sum(dim=1)): certainty = certainty + 1e-8
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * num, certainty.size(1)),
                                         replacement=False)
        # good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        good_matches = torch.gather(matches, 1, good_samples.unsqueeze(-1).expand(-1, -1, 4))
        good_certainty = torch.gather(certainty, 1, good_samples)
        return good_matches, good_certainty  # Hit this

    def forward(self, batch, batched=True):
        feature_pyramid = self.extract_backbone_features(batch, batched=batched)
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid)
        batch.update({"dense_corresps": dense_corresps})

    def forward_symmetric(self, batch, upsample=False, batched=True):
        feature_pyramid = self.extract_backbone_features(batch, upsample=upsample,
                                                         batched=batched)
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]))
            for scale, f_scale in feature_pyramid.items()
        }
        dense_corresps = self.decoder(f_q_pyramid, f_s_pyramid, upsample=upsample, **(
            batch["corresps"] if "corresps" in batch else {}))
        batch.update({"dense_corresps": dense_corresps})

    def to_pixel_coordinates(self, matches, H_A, W_A, H_B, W_B):
        kpts_A, kpts_B = matches[..., :2], matches[..., 2:]
        kpts_A = torch.stack(
            (W_A / 2 * (kpts_A[..., 0] + 1), H_A / 2 * (kpts_A[..., 1] + 1)), dim=-1)
        kpts_B = torch.stack(
            (W_B / 2 * (kpts_B[..., 0] + 1), H_B / 2 * (kpts_B[..., 1] + 1)), dim=-1)
        return kpts_A, kpts_B

    def match(self, batch):
        im1 = batch["color0"]
        im2 = batch["color1"]
        b, c, h, w = im1.shape
        b, c, h2, w2 = im2.shape
        hs, ws = self.h_resized, self.w_resized
        assert w == w2 == ws and h == h2 == hs, "For batched images we assume same size"

        finest_scale = 1

        self.forward(batch, batched=True)
        dense_corresps = batch["dense_corresps"]

        low_res_certainty = F.interpolate(
            dense_corresps[16]["dense_certainty"], size=(hs, ws),
            align_corners=True, mode="bilinear"
        )

        cert_clamp = 0
        factor = 0.5
        low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

        # Get certainty interpolation
        dense_certainty = dense_corresps[finest_scale]["dense_certainty"]
        dense_certainty = dense_certainty - low_res_certainty
        dense_certainty = dense_certainty.sigmoid()  # logits -> probs

        query_to_support = dense_corresps[finest_scale]["dense_flow"]
        query_to_support = query_to_support.permute(0, 2, 3, 1)
        if (query_to_support.abs() > 1).any():
            wrong = (query_to_support.abs() > 1).sum(dim=-1) > 0
            dense_certainty[wrong[:, None]] = 0
            dense_certainty[~batch['mask0_i'].unsqueeze(1)] = 0

        # Create im1 meshgrid
        query_coords = torch.meshgrid((
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=query_to_support.device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=query_to_support.device),
        ))
        query_coords = torch.stack((query_coords[1], query_coords[0]))
        query_coords = query_coords[None].expand(b, 2, hs, ws)
        query_coords = query_coords.permute(0, 2, 3, 1)

        query_to_support = torch.clamp(query_to_support, -1, 1)
        warp = torch.cat((query_coords, query_to_support), dim=-1)

        return warp, dense_certainty

    def inference(self, batch):
        num = 5000
        b = batch['image0'].size(0)
        dense_matches, dense_certainty = self.match(batch)
        sparse_matches = self.sample(dense_matches, dense_certainty, num)[0]

        h1, w1 = batch['hw0_i']
        kpts1 = sparse_matches[:, :, :2]
        kpts1 = torch.stack((w1 * (kpts1[:, :, 0] + 1) / 2,
                             h1 * (kpts1[:, :, 1] + 1) / 2,), dim=-1,)
        kpts1 *= batch['scale0'].unsqueeze(1)

        h2, w2 = batch['hw1_i']
        kpts2 = sparse_matches[:, :, 2:]
        kpts2 = torch.stack((w2 * (kpts2[:, :, 0] + 1) / 2,
                             h2 * (kpts2[:, :, 1] + 1) / 2,), dim=-1,)
        kpts2 *= batch['scale1'].unsqueeze(1)

        # b_ids = torch.zeros_like(kpts1[:, 0], device=kpts1.device).long()
        b_ids = torch.arange(b).unsqueeze(1).repeat(1, num).to(kpts1.device).reshape(-1)

        batch.update({
            'm_bids': b_ids,
            "mkpts0_f": kpts1.reshape(-1, 2),
            "mkpts1_f": kpts2.reshape(-1, 2),
        })
