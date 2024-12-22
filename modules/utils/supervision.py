import torch
from einops import repeat
from kornia.utils import create_meshgrid

from .geometry import warp_kpts


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
