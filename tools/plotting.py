import bisect
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from kornia.utils import create_meshgrid
from torch.nn.functional import grid_sample
from modules.utils.geometry import warp_kpts


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    else:
        thr = 1e-4
    return thr


# --- VISUALIZATION --- #

def dynamic_alpha(n_matches,
                  milestones=None,
                  alphas=None):
    if alphas is None:
        alphas = [1.0, 0.8, 0.4, 0.2]
    if milestones is None:
        milestones = [0, 300, 1000, 2000]
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
            milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert 1.0 >= alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1), 0, 1)


def fast_make_matching_figure(data, b_id):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr

    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches + 1e-8)

    alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    margin = 2
    H0, W0 = img0.shape
    H1, W1 = img1.shape
    H, W = margin + max(H0, H1), margin + W0 + margin + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[margin:H0 + margin, margin:W0 + margin] = img0
    out[margin:H1 + margin, margin + W0 + margin:margin + W0 + margin + W1] = img1
    out = np.stack([out] * 3, -1)

    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    return out


def fast_make_covision_figure(data, b_id):
    imsize = 640
    interpolate = partial(F.interpolate, size=[imsize, imsize], mode='nearest')
    mask_c0, mask_c1 = data['mask0'][b_id], data['mask1'][b_id]
    img0, img1 = data['color0'][b_id][None], data['color1'][b_id][None]  # (1, 3, rH, rW)
    img0, img1 = map(lambda x: interpolate(x)[0], [img0, img1])  # (3, imsize, imsize)
    pos0 = data['pos_mask0'][b_id].view(data['hw0_c'])  # (vsize, vsize)
    pos0 = interpolate(pos0.float()[None, None]).repeat(1, 3, 1, 1).bool().float()[0]
    pos1 = data['pos_mask1'][b_id].view(data['hw1_c'])  # (vsize, vsize)
    pos1 = interpolate(pos1.float()[None, None]).repeat(1, 3, 1, 1).bool().float()[0]
    pro0 = data['conf_c0'][b_id][:,1].view(data['hw0_c'])
    pro0 = interpolate(pro0[None, None]).repeat(1, 3, 1, 1)[0]
    pro1 = data['conf_c1'][b_id][:,1].view(data['hw1_c'])
    pro1 = interpolate(pro1[None, None]).repeat(1, 3, 1, 1)[0]

    seg0 = data['seg_c0'][b_id].view(data['hw0_c'])
    seg0 = interpolate(seg0.float()[None, None]).repeat(1, 3, 1, 1).bool().float()[0]
    seg1 = data['seg_c1'][b_id].view(data['hw1_c'])
    seg1 = interpolate(seg1.float()[None, None]).repeat(1, 3, 1, 1).bool().float()[0]

    grid = make_grid([img0, img1, pos0, pos1, seg0, seg1, pro0, pro1], nrow=2, pad_value=10)

    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    out = 255 * torch.ones(ndarr.shape, dtype=torch.uint8).numpy()
    out[:, :] = ndarr

    fingerprint = [
        'Dataset: {}'.format(data['dataset_name'][b_id]),
        'Scene ID: {}'.format(data['scene_id'][b_id]),
        'Pair ID: {}'.format(data['pair_id'][b_id]),
        'co-visible: {:.4f}/{:.4f}'.format(data['covisible0'][b_id],
                                           data['covisible1'][b_id]),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id].cpu().numpy(),
                                      data['imsize1'][b_id].cpu().numpy()),
        'Pair names: {}:{}'.format(data['pair_names'][0][b_id].split('/')[-1],
                                   data['pair_names'][1][b_id].split('/')[-1]),
    ]

    blcack = (0, 0, 0)
    white = (255, 255, 255)
    H = out.shape[0]
    sc = min(H / 640., 2.0)
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))),
                    cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, blcack, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))),
                    cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, white, 1, cv2.LINE_AA)

    return out


def fast_make_matching_covision_figure(data, b_id):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    gray0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    gray1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr

    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)

    alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h, w = max(h0, h1), max(w0, w1)
    H, W = margin * 3 + h * 2, margin * 3 + w * 2

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    interpolate0 = partial(F.interpolate, size=[h0,  w0], mode='nearest')
    interpolate1 = partial(F.interpolate, size=[h1,  w1], mode='nearest')
    wx = [margin, margin + w0, margin + w + margin, margin + w + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    hw0_c, hw1_c = data['hw0_c'], data['hw1_c']
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)

    # before outlier filtering
    sh = hx(row=2)
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    color = (np.array(color[:, :3]) * 255).astype(int)
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0+sh), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 2, c, -1, lineType=cv2.LINE_AA)

    # # after outlier filtering
    # sh = hx(row=3)
    # inliers = data['inliers'][b_id]
    # n_correct_r = np.sum(correct_mask[inliers])
    # precision_r = np.mean(correct_mask[inliers]) if len(correct_mask[inliers]) > 0 else 0
    # mkpts0, mkpts1 = np.round(kpts0).astype(int)[inliers], np.round(kpts1).astype(int)[inliers]
    # color = (np.array(color[:, :3]) * 255).astype(int)[inliers]
    # for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
    #     c = c.tolist()
    #     cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
    #     # display line end-points as circles
    #     cv2.circle(out, (x0, y0+sh), 2, c, -1, lineType=cv2.LINE_AA)
    #     cv2.circle(out, (x1 + margin + w, y1+sh), 2, c, -1, lineType=cv2.LINE_AA)

    # Big text.
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        # f'Precision({conf_thr:.2e}) ({100 * precision_r:.1f}%): {n_correct_r}/{len(mkpts0)}',
    ]
    sc = min(H / 640., 1.0)
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    fingerprint = [
        'Dataset: {}'.format(data['dataset_name'][b_id]),
        'Scene ID: {}'.format(data['scene_id'][b_id]),
        'Pair ID: {}'.format(data['pair_id'][b_id]),
        'co-visible: {:.4f}/{:.4f}'.format(data['covisible0'][b_id],
                                           data['covisible1'][b_id]),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id].cpu().numpy(),
                                      data['imsize1'][b_id].cpu().numpy()),
        'Pair names: {}:{}'.format(data['pair_names'][0][b_id].split('/')[-1],
                                   data['pair_names'][1][b_id].split('/')[-1]),
        'Rand Scale: {} - {}'.format(data['rands0'][b_id],
                                           data['rands1'][b_id]),
        'Fliped: {} - {}'.format(data['hflip0'][b_id],
                                 data['hflip1'][b_id]),
    ]
    sc = min(H / 640., 1.0)
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    return out


def fast_make_transform_figure(data, b_id):
    device = data['image0'].device
    b_mask = data['m_bids'] == b_id
    color2numpy = lambda x:  (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    color0 = data['color0'][b_id][None]  # (1, 3, rH, rW)
    color1 = data['color1'][b_id][None]  # (1, 3, rH, rW)
    scale0 = data['scale0'][b_id][None, None]  # (1, 1, 2)
    scale1 = data['scale1'][b_id][None, None]  # (1, 1, 2)
    imsize0 = data['imsize0'][b_id]  # (bs, 2) - (H, W)
    imsize1 = data['imsize1'][b_id]  # (bs, 2) - (H, W)

    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h, w = max(h0, h1), max(w0, w1)
    H, W = margin * 4 + h * 3, margin * 3 + w * 2

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w + margin, margin + w + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    hw0_c, hw1_c = data['hw0_c'], data['hw1_c']
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color2numpy(color0)
    out[sh: sh + h1, wx[2]: wx[3]] = color2numpy(color1)

    grid_pt0 = scale0 * create_meshgrid(h0, w0, False, device).reshape(h0*w0, 2)[None]  # (1, h0*w0, 2) - (x,y)
    grid_pt1 = scale1 * create_meshgrid(h1, w1, False, device).reshape(h1*w1, 2)[None]  # (1, h1*w1, 2) - (x,y)

    sh = hx(row=2)
    w_pt0 = warp_kpts(grid_pt0, data['depth0'][b_id][None], data['depth1'][b_id][None], data['T_0to1'][b_id][None], data['K0'][b_id][None], data['K1'][b_id][None], data['K0_'][b_id][None], data['K1_'][b_id][None], data['imsize1'][b_id][None])[-1]  # (1, h0*w0, 2) - (x,y)
    w_pt1 = warp_kpts(grid_pt1, data['depth1'][b_id][None], data['depth0'][b_id][None], data['T_1to0'][b_id][None], data['K1'][b_id][None], data['K0'][b_id][None], data['K1_'][b_id][None], data['K0_'][b_id][None], data['imsize0'][b_id][None])[-1]  # (1, h1*w1, 2) - (x,y)
    w_gd0 = ((w_pt0 / scale0) / w_pt0.new_tensor(data['hw1_i'])[[1, 0]][None, None]) * 2 - 1
    w_gd1 = ((w_pt1 / scale1) / w_pt1.new_tensor(data['hw0_i'])[[1, 0]][None, None]) * 2 - 1
    warp1 = grid_sample(color1, w_gd0.view(1, h0, w0, 2))
    warp0 = grid_sample(color0, w_gd1.view(1, h1, w1, 2))

    def zero_depth(depth, scale, h, w, oh, ow):
        # H, W = depth.shape
        scale = scale.squeeze()
        E = max(oh, ow)
        depth = depth[:E, :E]
        depth = depth[None, None]
        depth = F.interpolate(depth, (h, w))
        depth = (depth == 0).repeat(1, 3, 1, 1)
        return depth

    zero_depth0 = zero_depth(data['depth0'][b_id], scale0, h0, w0, imsize0[1], imsize0[0])
    zero_depth1 = zero_depth(data['depth1'][b_id], scale1, h1, w1, imsize1[1], imsize1[0])
    warp1[zero_depth0] = 0
    warp0[zero_depth1] = 0
    out[sh: sh + h0, wx[0]: wx[1]] = color2numpy(warp1)
    out[sh: sh + h1, wx[2]: wx[3]] = color2numpy(warp0)

    sh = hx(row=3)
    Rot = grid_pt0.new_tensor(data['Rot'][b_id])
    Tns = grid_pt0.new_tensor(data['Tns'][b_id])
    T_0to1 = data['T_0to1'][b_id].clone()
    T_0to1[:3, :3] = Rot
    T_0to1[:3, 3] = Tns
    Rot1 = grid_pt1.new_tensor(data['Rot1'][b_id])
    Tns1 = grid_pt1.new_tensor(data['Tns1'][b_id])
    T_1to0 = data['T_1to0'][b_id].clone()
    T_1to0[:3, :3] = Rot1
    T_1to0[:3, 3] = Tns1
    w_pt0 = warp_kpts(grid_pt0, data['depth0'][b_id][None], data['depth1'][b_id][None], T_0to1[None], data['K0'][b_id][None], data['K1'][b_id][None], data['imsize1'][b_id][None])[-1]  # (1, h0*w0, 2) - (x,y)
    w_pt1 = warp_kpts(grid_pt1, data['depth1'][b_id][None], data['depth0'][b_id][None], T_1to0[None], data['K1'][b_id][None], data['K0'][b_id][None], data['imsize0'][b_id][None])[-1]  # (1, h1*w1, 2) - (x,y)
    w_gd0 = (w_pt0 / w_pt0.new_tensor(data['hw1_i'])[[1, 0]][None, None]) * 2 - 1
    w_gd1 = (w_pt1 / w_pt1.new_tensor(data['hw0_i'])[[1, 0]][None, None]) * 2 - 1
    warp1 = grid_sample(color1, w_gd0.view(1, h0, w0, 2))
    warp0 = grid_sample(color0, w_gd1.view(1, h1, w1, 2))
    warp1[zero_depth0] = 0
    warp0[zero_depth1] = 0
    out[sh: sh + h0, wx[0]: wx[1]] = color2numpy(warp1)
    out[sh: sh + h1, wx[2]: wx[3]] = color2numpy(warp0)
    # out[sh: sh + h0, wx[0]: wx[1]] = ((data['depth0'][b_id][:h0, :w0] == 0).float()[:, :, None].repeat(1, 1, 3) * 255).cpu().numpy()
    # out[sh: sh + h1, wx[2]: wx[3]] = ((data['depth1'][b_id][:h0, :w0] == 0).float()[:, :, None].repeat(1, 1, 3) * 255).cpu().numpy()

    # Rot and Tns error
    delta, deg = 'Delta ', ' deg'
    err_t = data['t_errs'][b_id]
    err_R = data['R_errs'][b_id]
    e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
    e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)

    # Big text.
    text = [
        f'{delta}R: {e_R}   {delta}t: {e_t}',
    ]
    sc = min(H / 640., 1.0)
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    fingerprint = [
        'Dataset: {}'.format(data['dataset_name'][b_id]),
        'Scene ID: {}'.format(data['scene_id'][b_id]),
        'Pair ID: {}'.format(data['pair_id'][b_id]),
        'co-visible: {:.4f}/{:.4f}'.format(data['covisible0'][b_id],
                                           data['covisible1'][b_id]),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id].cpu().numpy(),
                                      data['imsize1'][b_id].cpu().numpy()),
        'Pair names: {}:{}'.format(data['pair_names'][0][b_id].split('/')[-1],
                                   data['pair_names'][1][b_id].split('/')[-1]),
    ]
    sc = min(H / 640., 1.0)
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_fg, 1, cv2.LINE_AA)
    # cv2.imwrite('out.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    return out


def project_kpts_to_camera(data, b_id, title, dir):
    b_mask = data['m_bids'] == b_id
    kpts0 = data['mkpts0_f'][b_mask]
    inliers = data['inliers'][b_id]
    kpts0 = kpts0[inliers]
    # kpts1 = data['mkpts1_f'][b_mask]

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id]
        # kpts1 = kpts1 / data['scale1'][b_id]

    def project(kpts, K, depth):
        kpts_long = kpts.round().long()

        # Sample depth, get calculable_mask on depth != 0
        kpts_depth = depth[kpts_long[:, 1], kpts_long[:, 0]]
        nonzero_mask = kpts_depth != 0

        kpts_hom = torch.cat([kpts, torch.ones_like(kpts[:, [0]])], dim=-1) * kpts_depth[..., None]
        kpts_cam = K.inverse() @ kpts_hom.transpose(0, 1)  # (3, L)
        return kpts_cam.transpose(0, 1)[nonzero_mask]

    kptc0 = project(kpts0, data['K0'][b_id], data['depth0'][b_id])
    # kptc1 = project(kpts1, data['K1'][b_id], data['depth1'][b_id])

    # import numpy as np
    # np.savetxt('kptc0.txt', kptc0.cpu().numpy())
    # np.savetxt('kptc1.txt', kptc1.cpu().numpy())
    path = dir / f'{title}_C_K.txt'
    np.savetxt(str(path), kptc0.cpu().numpy())


def fast_make_matching_robust_fitting_figure(data, b_id):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    gray0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    gray1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr

    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)

    alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h, w = max(h0, h1), max(w0, w1)
    H, W = margin * 4 + h * 3, margin * 3 + w * 2

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    interpolate0 = partial(F.interpolate, size=[h0,  w0], mode='nearest')
    interpolate1 = partial(F.interpolate, size=[h1,  w1], mode='nearest')
    wx = [margin, margin + w0, margin + w + margin, margin + w + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    # before outlier filtering
    sh = hx(row=2)
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
    color_v = (np.array(color[:, :3]) * 255).astype(int)
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color_v):
        c = c.tolist()
        cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0+sh), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 2, c, -1, lineType=cv2.LINE_AA)

    # after outlier filtering
    sh = hx(row=3)
    inliers = data['inliers'][b_id]
    n_correct_r = np.sum(correct_mask[inliers])
    precision_r = np.mean(correct_mask[inliers]) if len(correct_mask[inliers]) > 0 else 0
    mkpts0, mkpts1 = np.round(kpts0).astype(int)[inliers], np.round(kpts1).astype(int)[inliers]
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
    color_v = (np.array(color[:, :3]) * 255).astype(int)[inliers]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color_v):
        c = c.tolist()
        cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0+sh), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 2, c, -1, lineType=cv2.LINE_AA)

    # Rot and Tns error
    delta, deg = 'Delta ', ' deg'
    err_t = data['t_errs'][b_id]
    err_R = data['R_errs'][b_id]
    # err_t2 = data['t_errs2'][b_id]
    e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
    e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
    # e_t2 = 'FAIL' if np.isinf(err_t2) else '{:.1f}{}'.format(err_t2, ' m')

    # Big text.
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision_r:.1f}%): {n_correct_r}/{len(mkpts0)}',
        f'{delta}R: {e_R}   {delta}t: {e_t}',
        # f'{delta}T: {e_t2}',
    ]
    sc = min(H / 640., 1.0)
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    fingerprint = [
        'Dataset: {}'.format(data['dataset_name'][b_id]),
        'Scene ID: {}'.format(data['scene_id'][b_id]),
        'Pair ID: {}'.format(data['pair_id'][b_id]),
        'co-visible: {:.4f}/{:.4f}'.format(data['covisible0'][b_id],
                                           data['covisible1'][b_id]),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id].cpu().numpy(),
                                      data['imsize1'][b_id].cpu().numpy()),
        'Pair names: {}:{}'.format(data['pair_names'][0][b_id].split('/')[-1],
                                   data['pair_names'][1][b_id].split('/')[-1]),
        'Rand Scale: {} - {}'.format(data['rands0'][b_id],
                                           data['rands1'][b_id]),
        'Fliped: {} - {}'.format(data['hflip0'][b_id],
                                 data['hflip1'][b_id]),
    ]
    sc = min(H / 640., 1.0)
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    return out


def fast_make_matching_robust_fitting_figure_without_label(data, b_id):
    robust_fitting = True if 'inliers' in list(data.keys()) and data['inliers'] is not None else False
    b_mask = data['m_bids'] == b_id

    gray0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    gray1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    rows = 2 if not robust_fitting else 3
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h, w = max(h0, h1), max(w0, w1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w * 2

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w + margin, margin + w + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    # before outlier filtering
    sh = hx(row=2)
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        c = (159, 212, 252)
        cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        # display line end-points as circles
        c = (230, 216, 132)
        cv2.circle(out, (x0, y0+sh), 1, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 1, c, -1, lineType=cv2.LINE_AA)

    # after outlier filtering
    if robust_fitting:
        sh = hx(row=3)
        inliers = data['inliers'][b_id]
        mkpts0, mkpts1 = np.round(kpts0).astype(int)[inliers], np.round(kpts1).astype(int)[inliers]
        out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
        out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            c = (0, 255, 0)
            cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
            # display line end-points as circles
            cv2.circle(out, (x0, y0+sh), 1, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + margin + w, y1+sh), 1, c, -1, lineType=cv2.LINE_AA)

    # Big text.
    text = [
        f' ',
        f'#Matches {len(kpts0)}',
        f'#Matches {sum(data["inliers"][b_id])}' if robust_fitting else '',
    ]
    sc = min(H / 640., 1.0)
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX, 1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    fingerprint = [
        'Dataset: {}'.format(data['dataset_name'][b_id]),
        'Scene ID: {}'.format(data['scene_id'][b_id]),
        'Pair ID: {}'.format(data['pair_id'][b_id]),
        'co-visible: {:.4f}/{:.4f}'.format(data['covisible0'][b_id],
                                           data['covisible1'][b_id]),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id].cpu().numpy(),
                                      data['imsize1'][b_id].cpu().numpy()),
        'Pair names: {}:{}'.format(data['pair_names'][0][b_id].split('/')[-1],
                                   data['pair_names'][1][b_id].split('/')[-1]),
        'Rand Scale: {} - {}'.format(data['rands0'][b_id],
                                     data['rands1'][b_id]),
        'Offset: {} - {}'.format(data['offset0'][b_id].cpu().numpy(),
                                 data['offset1'][b_id].cpu().numpy()),
        'Fliped: {} - {}'.format(data['hflip0'][b_id],
                                 data['hflip1'][b_id]),
    ]
    sc = min(H / 1280., 1.0)
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    return out
