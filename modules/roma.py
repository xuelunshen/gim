# -*- coding: utf-8 -*-
# @Author  : Parskatt

import math
import torch
import warnings
import torch.nn.functional as F
import torchvision.models as tvm

from torch import nn
from einops import rearrange

# noinspection PyPackages
from .dino import vit_large, Block, MemEffAttention

resolutions = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


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
        predict_features=False,
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
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        with warnings.catch_warnings():
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(
                cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1
            )
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats


class VGG19(nn.Module):
    def __init__(self, pretrained=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])

    def forward(self, x, **kwargs):
        feats = {}
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats[scale] = x
                scale = scale * 2
            x = layer(x)
        return feats


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_decoder,
        gps,
        proj,
        conv_refiner,
        detach=False,
        scales="all",
        pos_embeddings=None,
        num_refinement_steps_per_scale=1,
        warp_noise_std=0.0,
        displacement_dropout_p=0.0,
        gm_warp_dropout_p=0.0,
        flow_upsample_mode="bilinear",
    ):
        super().__init__()
        self.embedding_decoder = embedding_decoder
        self.num_refinement_steps_per_scale = num_refinement_steps_per_scale
        self.gps = gps
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.detach = detach
        if pos_embeddings is None:
            self.pos_embeddings = {}
        else:
            self.pos_embeddings = pos_embeddings
        if scales == "all":
            self.scales = ["32", "16", "8", "4", "2", "1"]
        else:
            self.scales = scales
        self.warp_noise_std = warp_noise_std
        self.refine_init = 4
        self.displacement_dropout_p = displacement_dropout_p
        self.gm_warp_dropout_p = gm_warp_dropout_p
        self.flow_upsample_mode = flow_upsample_mode

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

    def get_positional_embedding(self, b, h, w, device):
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
        coarse_embedded_coords = self.pos_embedding(coarse_coords)
        return coarse_embedded_coords

    def forward(
        self,
        f1,
        f2,
        gt_warp=None,
        gt_prob=None,
        upsample=False,
        flow=None,
        certainty=None,
        scale_factor=1,
    ):
        coarse_scales = self.embedding_decoder.scales()
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"]
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])
        old_stuff = torch.zeros(
            b,
            self.embedding_decoder.hidden_dim,
            *sizes[coarsest_scale],
            device=f1[coarsest_scale].device,
        )
        corresps = {}
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = 0.0
        else:
            flow = F.interpolate(
                flow,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
            certainty = F.interpolate(
                certainty,
                size=sizes[coarsest_scale],
                align_corners=False,
                mode="bilinear",
            )
        displacement = 0.0
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]
            if new_scale in self.proj:
                f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            if ins in coarse_scales:
                old_stuff = F.interpolate(
                    old_stuff, size=sizes[ins], mode="bilinear", align_corners=False
                )
                gp_posterior = self.gps[new_scale](f1_s, f2_s)
                gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )

                if self.embedding_decoder.is_classifier:
                    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0, 3, 1, 2)
                    (
                        corresps[ins].update(
                            {
                                "gm_cls": gm_warp_or_cls,
                                "gm_certainty": certainty,
                            }
                        )
                        # if self.training
                        # else None
                    )
                else:
                    (
                        corresps[ins].update(
                            {
                                "gm_flow": gm_warp_or_cls,
                                "gm_certainty": certainty,
                            }
                        )
                        if self.training
                        else None
                    )
                    flow = gm_warp_or_cls.detach()

            if new_scale in self.conv_refiner:
                (
                    corresps[ins].update({"flow_pre_delta": flow})
                    # if self.training
                    # else None
                )
                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s,
                    f2_s,
                    flow,
                    scale_factor=scale_factor,
                    logits=certainty,
                )
                # (
                #     corresps[ins].update(
                #         {
                #             "delta_flow": delta_flow,
                #         }
                #     )
                #     # if self.training
                #     # else None
                # )
                displacement = ins * torch.stack(
                    (
                        delta_flow[:, 0].float() / (self.refine_init * w),
                        delta_flow[:, 1].float() / (self.refine_init * h),
                    ),
                    dim=1,
                )
                flow = flow + displacement
                certainty = (
                    certainty + delta_certainty
                )  # predict both certainty and displacement
            corresps[ins].update(
                {
                    "certainty": certainty,
                    "flow": flow,
                }
            )
            if new_scale != "1":
                flow = F.interpolate(
                    flow,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                certainty = F.interpolate(
                    certainty,
                    size=sizes[ins // 2],
                    mode=self.flow_upsample_mode,
                )
                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()
        return corresps


class ResNet50(nn.Module):
    def __init__(
        self,
        pretrained=False,
        high_res=False,
        weights=None,
        dilation=None,
        freeze_bn=True,
        anti_aliased=False,
        early_exit=False,
    ) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False, False, False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(
                    weights=weights, replace_stride_with_dilation=dilation
                )
            else:
                self.net = tvm.resnet50(
                    pretrained=pretrained, replace_stride_with_dilation=dilation
                )

        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit

    def forward(self, x, **kwargs):
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
        if self.early_exit:
            return feats
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
        displacement_emb=None,
        displacement_emb_dim=None,
        local_corr_radius=None,
        corr_in_other=None,
        no_im_B_fm=False,
        concat_logits=False,
        use_bias_block_1=True,
        use_cosine_corr=False,
        disable_local_corr_grad=False,
        is_classifier=False,
        sample_mode="bilinear",
        norm_type=nn.BatchNorm2d,
        bn_momentum=0.1,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim,
            hidden_dim,
            dw=dw,
            kernel_size=kernel_size,
            bias=use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2, displacement_emb_dim, 1, 1, 0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_radius
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode

    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias=True,
        norm_type=nn.BatchNorm2d,
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
            bias=bias,
        )
        # noinspection PyArgumentList
        norm = (
            norm_type(out_dim, momentum=self.bn_momentum)
            if norm_type is nn.BatchNorm2d
            else norm_type(num_channels=out_dim)
        )
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(self, x, y, flow, scale_factor=1, logits=None):
        b, c, hs, ws = x.shape
        with torch.no_grad():
            x_hat = F.grid_sample(
                y,
                flow.permute(0, 2, 3, 1),
                align_corners=False,
                mode=self.sample_mode,
            )
        if self.has_displacement_emb:
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=x.device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=x.device),
                )
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            in_displacement = flow - im_A_coords
            emb_in_displacement = self.disp_emb(
                40 / 32 * scale_factor * in_displacement
            )
            if self.local_corr_radius:
                if self.corr_in_other:
                    # Corr in other means take a kxk grid around the predicted coordinate in other image
                    local_corr = local_correlation(
                        x,
                        y,
                        local_radius=self.local_corr_radius,
                        flow=flow,
                        sample_mode=self.sample_mode,
                    )
                else:
                    raise NotImplementedError(
                        "Local corr in own frame should not be used."
                    )
                if self.no_im_B_fm:
                    x_hat = torch.zeros_like(x)
                d = torch.cat((x, x_hat, emb_in_displacement, local_corr), dim=1)
            else:
                d = torch.cat((x, x_hat, emb_in_displacement), dim=1)
        else:
            if self.no_im_B_fm:
                x_hat = torch.zeros_like(x)
            d = torch.cat((x, x_hat), dim=1)
        if self.concat_logits:
            d = torch.cat((d, logits), dim=1)
        d = self.block1(d)
        d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :-1], d[:, -1:]
        return displacement, certainty


class CNNandDinov2(nn.Module):
    def __init__(
        self,
        cnn_kwargs=None,
        use_vgg=False,
        dinov2_weights=None,
    ):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                map_location="cpu",
            )

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.dinov2_vitl14 = [dinov2_vitl14]  # ugly hack to not show parameters to DDP

    def train(self, mode: bool = True):
        return self.cnn.train(mode)

    def forward(self, x, upsample=False):
        B, C, H, W = x.shape
        feature_pyramid = self.cnn(x)

        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x)
                features_16 = (
                    dinov2_features_16["x_norm_patchtokens"]
                    .permute(0, 2, 1)
                    .reshape(B, 1024, H // 14, W // 14)
                )
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid


class RegressionMatcher(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode="threshold",
        upsample_preds=False,
        symmetric=False,
        name=None,
        attenuate_cert=None,
        recrop_upsample=False,
    ):
        super().__init__()
        self.attenuate_cert = attenuate_cert
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.w_resized = w
        self.h_resized = h
        self.sample_mode = sample_mode
        self.upsample_preds = upsample_preds
        self.upsample_res = (14 * 16 * 6, 14 * 16 * 6)
        self.symmetric = symmetric
        self.sample_thresh = 0.05
        self.recrop_upsample = recrop_upsample

    def get_output_resolution(self):
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res

    def extract_backbone_features(self, batch, batched=True, upsample=False):
        x_q = batch["color0"]
        x_s = batch["color1"]
        if batched:
            X = torch.cat((x_q, x_s), dim=0)
            feature_pyramid = self.encoder(X, upsample=upsample)
        else:
            feature_pyramid = self.encoder(x_q, upsample=upsample), self.encoder(
                x_s, upsample=upsample
            )
        return feature_pyramid

    def sample(
            self,
            dense_matches,
            dense_certainty,
            num=10000,
    ):
        b = dense_certainty.size(0)
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            dense_certainty = dense_certainty.clone()
            dense_certainty[dense_certainty > upper_thresh] = 1
        matches, certainty = (
            dense_matches.reshape(b, -1, 4),
            dense_certainty.reshape(b, -1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        if not all(certainty.sum(dim=1)): certainty = certainty + 1e-8
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * num, certainty.size(1)),
                                         replacement=False)
        # good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        good_matches = torch.gather(matches, 1, good_samples.unsqueeze(-1).expand(-1, -1, 4))
        good_certainty = torch.gather(certainty, 1, good_samples)
        return good_matches, good_certainty

    def forward(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        if batched:
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )

        batch.update({"corresps": corresps})

    def forward_symmetric(self, batch, batched=True, upsample=False, scale_factor=1):
        feature_pyramid = self.extract_backbone_features(
            batch, batched=batched, upsample=upsample
        )
        f_q_pyramid = feature_pyramid
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim=0)
            for scale, f_scale in feature_pyramid.items()
        }
        corresps = self.decoder(
            f_q_pyramid,
            f_s_pyramid,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )
        return corresps

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        kpts_A = torch.stack(
            (W_A / 2 * (kpts_A[..., 0] + 1), H_A / 2 * (kpts_A[..., 1] + 1)), dim=-1
        )
        kpts_B = torch.stack(
            (W_B / 2 * (kpts_B[..., 0] + 1), H_B / 2 * (kpts_B[..., 1] + 1)), dim=-1
        )
        return kpts_A, kpts_B

    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        kpts_A = torch.stack(
            (2 / W_A * kpts_A[..., 0] - 1, 2 / H_A * kpts_A[..., 1] - 1), dim=-1
        )
        kpts_B = torch.stack(
            (2 / W_B * kpts_B[..., 0] - 1, 2 / H_B * kpts_B[..., 1] - 1), dim=-1
        )
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple=True, return_inds=False):
        x_A_to_B = F.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0].mT
        cert_A_to_B = F.grid_sample(
            certainty[None, None, ...],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
        )[0, 0, 0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (cert_A_to_B[:, None] > self.sample_thresh),
            as_tuple=True,
        )

        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)

    def recrop(self, certainty, image_path):
        return None

    def match(self, batch):
        im1 = batch["color0"]
        im2 = batch["color1"]
        b, c, h, w = im1.shape
        b, c, h2, w2 = im2.shape
        hs, ws = self.h_resized, self.w_resized
        assert w == w2 == ws and h == h2 == hs, "For batched images we assume same size"

        finest_scale = 1
        corresps = batch["corresps"]

        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws),
            align_corners=False, mode="bilinear"
        )

        cert_clamp = 0
        factor = 0.5
        low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

        certainty = corresps[finest_scale]["certainty"]
        certainty = certainty - low_res_certainty
        certainty = certainty.sigmoid()  # logits -> probs

        im_A_to_im_B = corresps[finest_scale]["flow"]
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any():
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
            certainty[~batch['mask0_i'].unsqueeze(1)] = 0

        # Create im_A meshgrid
        im_A_coords = torch.meshgrid((
            torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=im_A_to_im_B.device),
            torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=im_A_to_im_B.device),
        ))
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)

        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)

        return warp, certainty

    def inference(self, batch):
        num = 5000
        b = batch['image0'].size(0)
        dense_matches, dense_certainty = self.match(batch)
        sparse_matches = self.sample(dense_matches, dense_certainty, num)[0]

        batch.update({
            'hw0_i': batch['image0'].shape[2:],
            'hw1_i': batch['image1'].shape[2:]
        })

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


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        blocks,
        hidden_dim,
        out_dim,
        is_classifier=False,
        pos_enc=True,
        learned_embeddings=False,
        embedding_dim=None,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [16]
        self.is_classifier = is_classifier
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        if self.learned_embeddings:
            self.learned_pos_embeddings = nn.Parameter(
                nn.init.kaiming_normal_(
                    torch.empty((1, hidden_dim, embedding_dim, embedding_dim))
                )
            )

    def scales(self):
        return self._scales.copy()

    def forward(self, gp_posterior, features, old_stuff, new_scale):
        def get_grid(b, h, w, device):
            grid = torch.meshgrid(
                *[
                    torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
                    for n in (b, h, w)
                ]
            )
            grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
            return grid

        B, C, H, W = gp_posterior.shape
        x = torch.cat((gp_posterior, features), dim=1)
        B, C, H, W = x.shape
        grid = get_grid(B, H, W, x.device).reshape(B, H * W, 2)
        if self.learned_embeddings:
            pos_enc = (
                F.interpolate(
                    self.learned_pos_embeddings,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .permute(0, 2, 3, 1)
                .reshape(1, H * W, C)
            )
        else:
            pos_enc = 0
        tokens = x.reshape(B, C, H * W).permute(0, 2, 1) + pos_enc
        z = self.blocks(tokens)
        out = self.to_out(z)
        out = out.permute(0, 2, 1).reshape(B, self.out_dim, H, W)
        warp, certainty = out[:, :-1], out[:, -1:]
        return warp, certainty, None


def kde(x, std=0.1):
    # use a gaussian kernel to estimate density
    x = x.half()  # Do it in half precision TODO: remove hardcoding
    scores = (-torch.cdist(x, x) ** 2 / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density


def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow=None,
    sample_mode="bilinear",
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    corr = torch.empty((B, K, h, w), device=feature0.device, dtype=feature0.dtype)
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=feature0.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=feature0.device),
            )
        )
        coords = torch.stack((coords[1], coords[0]), dim=-1)[None].expand(B, h, w, 2)
    else:
        coords = flow.permute(0, 2, 3, 1)  # If using flow, sample around flow target.
    local_window = torch.meshgrid(
        (
            torch.linspace(
                -2 * local_radius / h,
                2 * local_radius / h,
                2 * r + 1,
                device=feature0.device,
            ),
            torch.linspace(
                -2 * local_radius / w,
                2 * local_radius / w,
                2 * r + 1,
                device=feature0.device,
            ),
        )
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * r + 1, 2 * r + 1, 2)
        .reshape(1, (2 * r + 1) ** 2, 2)
    )
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (
                coords[_, :, :, None] + local_window[:, None, None]
            ).reshape(1, h, w * (2 * r + 1) ** 2, 2)
            window_feature = F.grid_sample(
                feature1[_ : _ + 1],
                local_window_coords,
                padding_mode=padding_mode,
                align_corners=False,
                mode=sample_mode,  #
            )
            window_feature = window_feature.reshape(c, h, w, (2 * r + 1) ** 2)
        corr[_] = (
            (feature0[_, ..., None] / (c**0.5) * window_feature)
            .sum(dim=0)
            .permute(2, 0, 1)
        )
    return corr


@torch.no_grad()
def cls_to_flow_refine(cls):
    B, C, H, W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / res, 1 - 1 / res, steps=res, device=device)
            for _ in range(2)
        ]
    )
    G = torch.stack([G[1], G[0]], dim=-1).reshape(C, 2)
    cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices

    index = (
        torch.stack((mode - 1, mode, mode + 1, mode - res, mode + res), dim=1)
        .clamp(0, C - 1)
        .long()
    )
    neighbours = torch.gather(cls, dim=1, index=index)[..., None]
    flow = (
        neighbours[:, 0] * G[index[:, 0]]
        + neighbours[:, 1] * G[index[:, 1]]
        + neighbours[:, 2] * G[index[:, 2]]
        + neighbours[:, 3] * G[index[:, 3]]
        + neighbours[:, 4] * G[index[:, 4]]
    )
    tot_prob = neighbours.sum(dim=1)
    flow = flow / tot_prob
    return flow


def get_model(img_size, pretrained_backbone=True, **kwargs):
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(
            *[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]
        ),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        pos_enc=False,
    )
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 * 512 + 128 + (2 * 7 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 * 512 + 64 + (2 * 3 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 * 256 + 32 + (2 * 2 + 1) ** 2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64 + 16,
                128 + 16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict(
        {
            "16": proj16,
            "8": proj8,
            "4": proj4,
            "2": proj2,
            "1": proj1,
        }
    )
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    decoder = Decoder(
        coordinate_decoder,
        gps,
        proj,
        conv_refiner,
        detach=True,
        scales=["16", "8", "4", "2", "1"],
        displacement_dropout_p=displacement_dropout_p,
        gm_warp_dropout_p=gm_warp_dropout_p,
    )
    assert img_size is not None
    assert isinstance(img_size, list)
    assert len(img_size) <= 2
    if len(img_size) == 1: img_size = img_size * 2
    h, w = img_size
    encoder = CNNandDinov2(
        cnn_kwargs=dict(pretrained=pretrained_backbone),
        use_vgg=True,
    )
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, **kwargs)
    return matcher
