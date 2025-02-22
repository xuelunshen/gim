import torch


def Project_3d_to_2d(pts, K, H, W):
    """

    Args:
        pts: (bs, 3, N) points in camera coordinates
        K: (bs, 12) fisheye intrinsics
        H: (bs, 1) height of the depth image
        W: (bs, 1) width of the depth image

    Returns:

    """
    fx = 0
    fy = 1
    cx = 2
    cy = 3
    k1 = 4
    k2 = 5
    p1 = 6
    p2 = 7
    k3 = 8
    k4 = 9
    sx1 = 10
    sy1 = 11

    pts = pts[:, :2].permute(0, 2, 1)  # (bs, N, 2)
    K = K.transpose(0, 1)[...,None]  # (12, bs, 1)

    r = (pts**2).sum(dim=-1).sqrt()  # (bs, N)
    theta = torch.arctan(r)  # (bs, N)
    pts = pts * (theta / (r + 1e-4))[...,None]  # (bs, N, 2)

    Tr = 1 + K[k1] * theta ** 2 + K[k2] * theta ** 4 + K[k3] * theta ** 6 + K[k4] * theta ** 8  # (bs, N)

    Ud, Vd = pts[:, :, 0], pts[:, :, 1]  # (bs, N)
    Un = Ud * Tr + 2 * K[p1] * Ud * Vd + K[p2] * (theta ** 2 + 2 * Ud ** 2) + K[sx1] * theta ** 2   # (bs, N)
    Vn = Vd * Tr + 2 * K[p2] * Ud * Vd + K[p1] * (theta ** 2 + 2 * Vd ** 2) + K[sy1] * theta ** 2   # (bs, N)

    U = K[fx] * Un + K[cx]
    V = K[fy] * Vn + K[cy]

    UV = torch.stack((U, V), dim=-1)  # (bs, N, 2)

    # (bs, N)
    valid = (UV[:, :, 0] >= 0) & (UV[:, :, 0] < W - 1) & \
            (UV[:, :, 1] >= 0) & (UV[:, :, 1] < H - 1)

    UV[~valid] = -1

    return UV


def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1)[:, None]
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, depth_.new_tensor(float("nan")))
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, size0, size1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    Args:
        kpts0 (torch.Tensor): [b, N, 2] - <x, y>,
        depth0 (torch.Tensor): [b, oH, oW],
        depth1 (torch.Tensor): [b, oH, oW],
        T_0to1 (torch.Tensor): [b, 4, 4],
        K0 (torch.Tensor): [b, 3, 3],
        K1 (torch.Tensor): [b, 3, 3],
        size0 (torch.Tensor): [b, 2], (h, w)
        size1 (torch.Tensor): [b, 2], (h, w)
    Returns:
        pos_mask (torch.Tensor): [b, N]
        neg_mask (torch.Tensor): [b, N]
        w_kpts0 (torch.Tensor): [b, N, 2] - <x, y>
    """
    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L) w/ last coordinate is 1

    kpts0_samples = [sample_depth(pt[None], dp[:h.item(), :w.item()][None]) for pt, dp, (h, w) in zip(kpts0, depth0, size0)]
    kpts0_depth = torch.concat([x[0] for x in kpts0_samples])
    kpts0_valid0 = torch.concat([x[1] for x in kpts0_samples])

    nonzero_mask = kpts0_depth > 0
    nonzero_mask = valid = nonzero_mask & (kpts0_depth == kpts0_depth)
    # nonzero_mask = nonzero_mask & ~bad_mask

    kpts0_cam *= kpts0_depth[:, None]  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]
    visible = w_kpts0_depth_computed > 0.0001

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)

    w_kpts0 = w_kpts0_h[:, :, :2] / w_kpts0_h[:, :, [2]]  # (N, L, 2)

    bad_mask = w_kpts0 != w_kpts0
    w_kpts0[bad_mask] = 0  # invalid
    nonzero_mask = nonzero_mask & ~bad_mask[..., 0] & ~bad_mask[..., 1]

    # Covisible Check
    covisible_mask = torch.stack([(pt[:, 0] > 0) * (pt[:, 0] < (hw[1]-1)) *
                                  (pt[:, 1] > 0) * (pt[:, 1] < (hw[0]-1))
                                  for pt,hw in zip(w_kpts0, size1)])
    pos_mask = nonzero_mask * covisible_mask * visible

    return w_kpts0, pos_mask, valid
