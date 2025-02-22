import torch
from einops import repeat
from kornia.utils import create_meshgrid

from .geometry import warp_kpts

IGNORE_FEATURE = -2
UNMATCHED_FEATURE = -1


def Flip(pts, hf, vf, w, h):
    """

    Args:
        pts: (bs, 6400, 2) - <x, y>
        hf: (bs,)
        vf: (bs,)
        w: [bs]
        h: [bs]

    Returns:

    """
    pts = pts.clone()
    for i, hv in enumerate(zip(hf, vf)):
        if hv[1]:
            pts[i, :, 1] = h[i] - pts[i, :, 1]
        if hv[0]:
            pts[i, :, 0] = w[i] - pts[i, :, 0]
    return pts


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, scale):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape # original image shape
    _, _, H1, W1 = data['image1'].shape # original image shape
    gt = data['gt']
    N = gt.sum()
    scale = scale[0] # scale: 8 # Coarse shrink scale
    scale0 = data['scale0'][:, None][gt]
    scale1 = data['scale1'][:, None][gt]
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1]) # coarse-level image shape

    Hq_aug = data['Hq_aug'][gt]  # (bs, 3, 3)
    Hq_ori = data['Hq_ori'][gt]  # (bs, 3, 3)

    offset0, offset1 = data['offset0'][:, None][gt], data['offset1'][:, None][gt]  # (bs, 1, 2) - <x, y>
    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_fc = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1) # (bs, h0*w0, 2) - (x,y)
    grid_pt0_fs = grid_pt0_fc * scale
    grid_pt0_rs = Flip(grid_pt0_fs, data['hflip0'][gt], data['vflip0'][gt], data['resize0'][:, 1][gt] - 1, data['resize0'][:, 0][gt] - 1)  # rectified flip
    grid_pt0_rs += offset0
    grid_pt0_i = grid_pt0_rs * scale0

    grid_pt1_fc = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_fs = grid_pt1_fc * scale

    # grid_pt1_fs is (bs, h1*w1, 2), change it to homography coordinates (bs, h1*w1, 3)
    grid_pt1_fsr = torch.cat([grid_pt1_fs, torch.ones_like(grid_pt1_fs[:, :, :1])], dim=-1)
    # inverse each item in Hq_aug, and then apply it to grid_pt1_fs
    Hq_aug_inv = torch.inverse(Hq_aug)
    grid_pt1_fs = torch.einsum('bij,bjk->bik', grid_pt1_fsr, Hq_aug_inv.transpose(1, 2))
    grid_pt1_fs = grid_pt1_fs[..., :2] / grid_pt1_fs[..., [2]]

    grid_pt1_rs = Flip(grid_pt1_fs, data['hflip1'][gt], data['vflip1'][gt], data['resize1'][:, 1][gt] - 1, data['resize1'][:, 0][gt] - 1)  # rectified flip
    grid_pt1_rs += offset1
    grid_pt1_i = grid_pt1_rs * scale1

    grid_pt1_fi = grid_pt1_fs * scale1
    grid_pt1_fi = grid_pt1_fc * scale * scale1

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    # mask all points in padded region to (0, 0)
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'][gt])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'][gt])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    pos_mask0, neg_mask0, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'][gt], data['depth1'][gt], data['T_0to1'][gt], data['K0'][gt], data['K1'][gt], data['K0_'][gt], data['K1_'][gt], data['imsize1'][gt])
    pos_mask1, neg_mask1, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'][gt], data['depth0'][gt], data['T_1to0'][gt], data['K1'][gt], data['K0'][gt], data['K1_'][gt], data['K0_'][gt], data['imsize0'][gt])

    if 'mask0' in data:
        mask_c0, mask_c1 = data['mask0'][gt].flatten(-2), data['mask1'][gt].flatten(-2)
        pos_mask0, pos_mask1 = pos_mask0 & mask_c0, pos_mask1 & mask_c1

    w_pt0_s, w_pt1_s = w_pt0_i / scale1, w_pt1_i / scale0
    w_pt0_s, w_pt1_s = w_pt0_s - offset1, w_pt1_s - offset0
    w_pt0_s = Flip(w_pt0_s, data['hflip1'][gt], data['vflip1'][gt], data['resize1'][:, 1][gt] - 1, data['resize1'][:, 0][gt] - 1)
    w_pt1_s = Flip(w_pt1_s, data['hflip0'][gt], data['vflip0'][gt], data['resize0'][:, 1][gt] - 1, data['resize0'][:, 0][gt] - 1)

    # w_pt0_s is (bs, h0*w0, 2), change it to homography coordinates (bs, h0*w0, 3)
    w_pt0_s = torch.cat([w_pt0_s, torch.ones_like(w_pt0_s[:, :, :1])], dim=-1)
    # Hq_aug is (bs, 3, 3), apply it to w_pt0_s
    w_pt0_s = torch.einsum('bij,bjk->bik', w_pt0_s, Hq_aug.transpose(1, 2))
    w_pt0_s = w_pt0_s[..., :2] / w_pt0_s[..., [2]]

    w_pt0_c, w_pt1_c = w_pt0_s / scale, w_pt1_s / scale
    data.update({'pos_mask0': pos_mask0, 'pos_mask1': pos_mask1,
                 'neg_mask0': neg_mask0, 'neg_mask1': neg_mask1})

    w_pt0_fi = w_pt0_i - offset1 * scale1
    w_pt0_fi = Flip(w_pt0_fi,data['hflip1'][gt], data['vflip1'][gt], data['resize1'][:, 1][gt] * data['scale1'][:, 0][gt] - 1, data['resize1'][:, 0][gt] * data['scale1'][:, 1][gt] - 1)
    # w_pt0_fi is (bs, h0*w0, 2), change it to homography coordinates (bs, h0*w0, 3)
    w_pt0_fi = torch.cat([w_pt0_fi, torch.ones_like(w_pt0_fi[:, :, :1])], dim=-1)
    # Hq_aug is (bs, 3, 3), apply it to w_pt0_fi
    w_pt0_fi = torch.einsum('bij,bjk->bik', w_pt0_fi, Hq_ori.transpose(1, 2))
    w_pt0_fi = w_pt0_fi[..., :2] / w_pt0_fi[..., [2]]

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1 # (bs, h0*w0)
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0 # (bs, h1*w1)

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0 # (bs, h0*w0)
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0 # (bs, h1*w1)

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1) # (bs, h0*w0)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device, dtype=torch.int)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = torch.ones(len(b_ids), device=device, dtype=torch.int)
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        # logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_fi,
        'spv_pt1_i': grid_pt1_fi
    })


@torch.no_grad()
def spvs_fine(data, scale, radius):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    radius = radius // 2  # 2
    scale = scale[1]  # 2
    gt = data['gt']

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']  # in gt sub batch

    # 3. compute gt
    scale = scale * data['scale1'][gt][b_ids] if 'scale1' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]
    # expec_f_gt = torch.cat([expec_f_gt, data['zs_expec_f_gt']], dim=0) if training else expec_f_gt
    data.update({"expec_f_gt": expec_f_gt})


@torch.no_grad()
def spvs(data):
    pass
    pos_th = 3
    neg_th = 5
    epi_th = 5
    cc_th = None

    device = data['image0'].device
    gt = data['gt']

    keypoints0, keypoints1 = data["keypoints0"][gt], data["keypoints1"][gt]
    scale0, scale1 = data['scale0'][:, None][gt], data['scale1'][:, None][gt]
    offset0, offset1 = data['offset0'][:, None][gt], data['offset1'][:, None][gt]

    # d0, valid0 = sample_depth(kp0, depth0)
    # d1, valid1 = sample_depth(kp1, depth1)
    grid_pt0_rs = Flip(keypoints0, data['hflip0'][gt], data['vflip0'][gt], data['resize0'][:, 1][gt] - 1, data['resize0'][:, 0][gt] - 1)  # rectified flip
    grid_pt0_rs += offset0
    kpts0 = grid_pt0_rs * scale0

    Hq_aug = data['Hq_aug'][gt]  # (bs, 3, 3)
    Hq_aug_inv = torch.inverse(Hq_aug)
    grid_pt1_fsr = torch.cat([keypoints1, torch.ones_like(keypoints1[:, :, :1])], dim=-1)
    grid_pt1_fs = torch.einsum('bij,bjk->bik', grid_pt1_fsr, Hq_aug_inv.transpose(1, 2))
    grid_pt1_fs = grid_pt1_fs[..., :2] / grid_pt1_fs[..., [2]]
    grid_pt1_rs = Flip(grid_pt1_fs, data['hflip1'][gt], data['vflip1'][gt], data['resize1'][:, 1][gt] - 1, data['resize1'][:, 0][gt] - 1)  # rectified flip
    grid_pt1_rs += offset1
    kpts1 = grid_pt1_rs * scale1

    num0, num1 = kpts0.size(1), kpts1.size(1)

    kpts0_transformed, visible0, valid0 = warp_kpts(kpts0, data['depth0'][gt],
                                                    data['depth1'][gt], data['T_0to1'][gt],
                                                    data['K0'][gt], data['K1'][gt],
                                                    data['imsize0'][gt], data['imsize1'][gt])
    kpts1_transformed, visible1, valid1 = warp_kpts(kpts1, data['depth1'][gt],
                                                    data['depth0'][gt], data['T_1to0'][gt],
                                                    data['K1'][gt], data['K0'][gt],
                                                    data['imsize1'][gt], data['imsize0'][gt])
    mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

    w_pt0_s, w_pt1_s = kpts0_transformed / scale1, kpts1_transformed / scale0
    w_pt0_s, w_pt1_s = w_pt0_s - offset1, w_pt1_s - offset0
    w_pt0_s = Flip(w_pt0_s, data['hflip1'][gt], data['vflip1'][gt], data['resize1'][:, 1][gt] - 1, data['resize1'][:, 0][gt] - 1)
    w_pt1_s = Flip(w_pt1_s, data['hflip0'][gt], data['vflip0'][gt], data['resize0'][:, 1][gt] - 1, data['resize0'][:, 0][gt] - 1)

    # w_pt0_s is (bs, h0*w0, 2), change it to homography coordinates (bs, h0*w0, 3)
    w_pt0_s = torch.cat([w_pt0_s, torch.ones_like(w_pt0_s[:, :, :1])], dim=-1)
    # Hq_aug is (bs, 3, 3), apply it to w_pt0_s
    w_pt0_s = torch.einsum('bij,bjk->bik', w_pt0_s, Hq_aug.transpose(1, 2))
    w_pt0_s = w_pt0_s[..., :2] / w_pt0_s[..., [2]]

    # build a distance matrix of size [... x M x N]
    dist0 = torch.sum((w_pt0_s.unsqueeze(-2) - keypoints1.unsqueeze(-3)) ** 2, -1)
    dist1 = torch.sum((keypoints0.unsqueeze(-2) - w_pt1_s.unsqueeze(-3)) ** 2, -1)
    dist = torch.max(dist0, dist1)
    inf = dist.new_tensor(float("inf"))
    dist = torch.where(mask_visible, dist, inf)

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices

    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()
    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1 & (dist < pos_th**2)

    negative0 = (dist0.min(-1).values > neg_th**2) & valid0
    negative1 = (dist1.min(-2).values > neg_th**2) & valid1

    # pack the indices of positive matches
    # if -1: unmatched point
    # if -2: ignore point
    unmatched = min0.new_tensor(UNMATCHED_FEATURE)
    ignore = min0.new_tensor(IGNORE_FEATURE)
    m0 = torch.where(positive.any(-1), min0, ignore)
    m1 = torch.where(positive.any(-2), min1, ignore)
    m0 = torch.where(negative0, unmatched, m0)
    m1 = torch.where(negative1, unmatched, m1)

    def skew_symmetric(v):
        """Create a skew-symmetric matrix from a (batched) vector of size (..., 3)."""
        z = torch.zeros_like(v[..., 0])
        M = torch.stack(
            [
                z,
                -v[..., 2],
                v[..., 1],
                v[..., 2],
                z,
                -v[..., 0],
                -v[..., 1],
                v[..., 0],
                z,
            ],
            dim=-1,
        ).reshape(v.shape[:-1] + (3, 3))
        return M

    def T_to_E(T):
        """Convert batched poses (..., 4, 4) to batched essential matrices."""
        return skew_symmetric(T[:, :3, 3]) @ T[:, :3, :3]

    import numpy as np

    def to_homogeneous(points):
        """Convert N-dimensional points to homogeneous coordinates.
        Args:
            points: torch.Tensor or numpy.ndarray with size (..., N).
        Returns:
            A torch.Tensor or numpy.ndarray with size (..., N+1).
        """
        if isinstance(points, torch.Tensor):
            pad = points.new_ones(points.shape[:-1] + (1,))
            return torch.cat([points, pad], dim=-1)
        elif isinstance(points, np.ndarray):
            pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
            return np.concatenate([points, pad], axis=-1)
        else:
            raise ValueError

    def sym_epipolar_distance_all(p0, p1, E, eps=1e-15):
        if p0.shape[-1] != 3:
            p0 = to_homogeneous(p0)
        if p1.shape[-1] != 3:
            p1 = to_homogeneous(p1)
        p1_E_p0 = torch.einsum("...mi,...ij,...nj->...nm", p1, E, p0).abs()
        E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
        Et_p1 = torch.einsum("...ij,...mi->...mj", E, p1)
        d0 = p1_E_p0 / (E_p0[..., None, 0] ** 2 + E_p0[..., None, 1] ** 2 + eps).sqrt()
        d1 = (
                p1_E_p0
                / (Et_p1[..., None, :, 0] ** 2 + Et_p1[..., None, :,
                                                 1] ** 2 + eps).sqrt()
        )
        return (d0 + d1) / 2

    F = (
        data['K1'][gt].inverse().transpose(-1, -2)
        @ T_to_E(data['T_0to1'][gt])
        @ data['K0'][gt].inverse()
    )
    epi_dist = sym_epipolar_distance_all(kpts0, kpts1, F)

    # Add some more unmatched points using epipolar geometry
    if epi_th is not None:
        mask_ignore = (m0.unsqueeze(-1) == ignore) & (m1.unsqueeze(-2) == ignore)
        epi_dist = torch.where(mask_ignore, epi_dist, inf)
        exclude0 = epi_dist.min(-1).values > neg_th
        exclude1 = epi_dist.min(-2).values > neg_th
        m0 = torch.where((~valid0) & exclude0, ignore.new_tensor(-1), m0)
        m1 = torch.where((~valid1) & exclude1, ignore.new_tensor(-1), m1)

    return {
        "assignment": positive,
        "reward": (dist < pos_th**2).float() - (epi_dist > neg_th).float(),
        "matches0": m0,
        "matches1": m1,
        "matching_scores0": (m0 > -1).float(),
        "matching_scores1": (m1 > -1).float(),
        # "depth_keypoints0": d0,
        # "depth_keypoints1": d1,
        # "proj_0to1": kp0_1,
        # "proj_1to0": kp1_0,
        "visible0": visible0,
        "visible1": visible1,
    }
