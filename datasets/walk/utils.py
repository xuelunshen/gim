# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import numpy as np


def fast_make_matching_robust_fitting_figure(data, b_id=0):
    robust_fitting = True if 'inliers' in list(data.keys()) and data['inliers'] is not None else False

    gray0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    gray1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data['mkpts0_f']
    kpts1 = data['mkpts1_f']

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    rows = 3
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

    # only show keypoints
    sh = hx(row=2)
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        # display line end-points as circles
        c = (230, 216, 132)
        cv2.circle(out, (x0, y0+sh), 1, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 1, c, -1, lineType=cv2.LINE_AA)

    # show keypoints and correspondences
    sh = hx(row=3)
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    out[sh: sh + h0, wx[0]: wx[1]] = np.stack([gray0] * 3, -1)
    out[sh: sh + h1, wx[2]: wx[3]] = np.stack([gray1] * 3, -1)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        c = (159, 212, 252)
        cv2.line(out, (x0, y0+sh), (x1 + margin + w, y1+sh), color=c, thickness=1, lineType=cv2.LINE_AA)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        # display line end-points as circles
        c = (230, 216, 132)
        cv2.circle(out, (x0, y0+sh), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w, y1+sh), 2, c, -1, lineType=cv2.LINE_AA)

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
        'co-visible: {:.4f}/{:.4f}'.format(data['covisible0'],
                                           data['covisible1']),
        'Image sizes: {} - {}'.format(data['imsize0'][b_id],
                                      data['imsize1'][b_id]),
        'Pair names: {}:{}'.format(data['pair_names'][0].split('/')[-1],
                                   data['pair_names'][1].split('/')[-1]),
        'Rand Scale: {} - {}'.format(data['rands0'],
                                     data['rands1']),
        'Offset: {} - {}'.format(data['offset0'].cpu().numpy(),
                                 data['offset1'].cpu().numpy()),
        'Fliped: {} - {}'.format(data['hflip0'],
                                 data['hflip1']),
    ]
    sc = min(H / 1280., 1.0)
    Ht = int(18 * sc)  # text height
    txt_color_fg = (255, 255, 255)  # white
    txt_color_bg = (0, 0, 0)  # black
    for i, t in enumerate(reversed(fingerprint)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_SIMPLEX, .5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    return out[h+margin:]


def eudist(a, b):
    aa = np.sum(a ** 2, axis=-1, keepdims=True)
    bb = np.sum(b ** 2, axis=-1, keepdims=True).T
    cc = a @ b.T
    dist = aa + bb - 2*cc
    return dist


def covision(kpts, size):
    return (kpts[:, 0].max() - kpts[:, 0].min()) * \
           (kpts[:, 1].max() - kpts[:, 1].min()) / \
           (size[0] * size[1] + 1e-8)


view = lambda x: x.view([('', x.dtype)] * x.shape[1])


def intersected(x, y):
    intersected = np.intersect1d(view(x), view(y))
    z = intersected.view(x.dtype).reshape(-1, x.shape[1])
    return z
