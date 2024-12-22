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


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, K0_, K1_, size):
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
        K0_ (torch.Tensor): [b, 12], (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1)
        K1_ (torch.Tensor): [b, 12], (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1)
        size
    Returns:
        pos_mask (torch.Tensor): [b, N]
        neg_mask (torch.Tensor): [b, N]
        w_kpts0 (torch.Tensor): [b, N, 2] - <x, y>
    """
    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L) w/ last coordinate is 1

    kpts0_long = torch.cat([
        kpts0[[i]] if k.sum() == 0 else
        Project_3d_to_2d(kpts0_cam[i][None], k[None], depth0.size(1), depth0.size(2))
        for i, k in enumerate(K0_)
    ], dim=0).round().long()  # (N, L, 2)

    bad_mask = torch.logical_or(
        torch.logical_or(kpts0_long[..., 0] < 0, kpts0_long[..., 0] >= depth0.size(2)),
        torch.logical_or(kpts0_long[..., 1] < 0, kpts0_long[..., 1] >= depth0.size(1))
    )
    kpts0_long[bad_mask] = 0

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)

    nonzero_mask = kpts0_depth != 0
    nonzero_mask = nonzero_mask & ~bad_mask

    kpts0_cam *= kpts0_depth[:, None]  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)

    w_kpts0 = torch.cat([
        w_kpts0_h[[i],:,:2] / (w_kpts0_h[[i],:,2][...,None] + 1e-4) if k.sum() == 0 else
        Project_3d_to_2d(w_kpts0_cam[[i],:2]/(w_kpts0_cam[[i],2][:,None]+1e-4), k[None],
                         depth1.size(1), depth1.size(2))
        for i, k in enumerate(K1_)
    ], dim=0)  # (N, L, 2)

    bad_mask = w_kpts0 != w_kpts0
    w_kpts0[bad_mask] = 0  # invalid
    nonzero_mask = nonzero_mask & ~bad_mask[..., 0] & ~bad_mask[..., 1]

    # Covisible Check
    covisible_mask = torch.stack([(pt[:, 0] > 0) * (pt[:, 0] < (hw[1]-1)) *
                                  (pt[:, 1] > 0) * (pt[:, 1] < (hw[0]-1))
                                  for pt,hw in zip(w_kpts0, size)])
    w_kpts0_long = w_kpts0.round().long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    pos_mask = nonzero_mask * covisible_mask * consistent_mask
    neg_mask = nonzero_mask * ~covisible_mask

    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    return pos_mask, neg_mask, w_kpts0
