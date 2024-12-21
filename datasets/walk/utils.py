# -*- coding: utf-8 -*-
# @Author  : xuelun

import math

import cv2
import torch
import random
import numpy as np

from albumentations.augmentations import functional as F

from datasets.utils import get_divisible_wh


def fast_make_matching_robust_fitting_figure(data, b_id=0, transpose=False):
    robust_fitting = True if 'inliers' in list(data.keys()) and data['inliers'] is not None else False

    gray0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    gray1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.uint8)
    kpts0 = data['mkpts0_f']
    kpts1 = data['mkpts1_f']

    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    if transpose:
        gray0 = cv2.rotate(gray0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray1 = cv2.rotate(gray1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        h0, w0 = data['hw0_i']
        h1, w1 = data['hw1_i']
        kpts0_new = np.copy(kpts0)
        kpts1_new = np.copy(kpts1)
        kpts0_new[:, 0], kpts0_new[:, 1] = kpts0[:, 1], w0 - kpts0[:, 0]
        kpts1_new[:, 0], kpts1_new[:, 1] = kpts1[:, 1], w1 - kpts1[:, 0]
        kpts0, kpts1 = kpts0_new, kpts1_new
        (h0, w0), (h1, w1) = (w0, h0), (w1, h1)
    else:
        (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']

    rows = 3
    margin = 2
    h, w = max(h0, h1), max(w0, w1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w * 2

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w + margin, margin + w + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    if transpose:
        color0 = cv2.rotate(color0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        color1 = cv2.rotate(color1, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        'Image sizes: {} - {}'.format(
            tuple(reversed(data['imsize0'][b_id])) if transpose and isinstance(data['imsize0'][b_id], (list, tuple, np.ndarray)) and len(data['imsize0'][b_id]) >= 2 else data['imsize0'][b_id],
            tuple(reversed(data['imsize1'][b_id])) if transpose and isinstance(data['imsize1'][b_id], (list, tuple, np.ndarray)) and len(data['imsize1'][b_id]) >= 2 else data['imsize1'][b_id]),
        'Pair names: {}:{}'.format(data['pair_names'][0].split('/')[-1],
                                   data['pair_names'][1].split('/')[-1]),
        'Rand Scale: {} - {}'.format(data['rands0'],
                                     data['rands1']),
        'Offset: {} - {}'.format(data['offset0'].cpu().numpy(),
                                 data['offset1'].cpu().numpy()),
        'Fliped: {} - {}'.format(data['hflip0'],
                                 data['hflip1']),
        'Transposed: {}'.format(transpose)
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
    intersected_ = np.intersect1d(view(x), view(y))
    z = intersected_.view(x.dtype).reshape(-1, x.shape[1])
    return z


def imread_color(path, augment_fn=None, read_size=None, source=None):
    if augment_fn is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if source is None else source
        image = cv2.resize(image, read_size) if read_size is not None else image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if source is None else image
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if source is None else source
        image = cv2.resize(image, read_size) if read_size is not None else image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if source is None else image
        image = augment_fn(image)
    return image  # (h, w)


def get_resized_wh(w, h, resize, aug_prob):
    nh, nw = resize
    sh, sw = nh / h, nw / w
    # scale = min(sh, sw)
    scale = random.choice([sh, sw]) if aug_prob != 1.0 else min(sh, sw)
    w_new, h_new = int(round(w*scale)), int(round(h*scale))
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size[0], pad_size[1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
    elif inp.ndim == 3:
        padded = np.zeros((pad_size[0], pad_size[1], inp.shape[-1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
    else:
        raise NotImplementedError()

    if ret_mask:
        mask = np.zeros((pad_size[0], pad_size[1]), dtype=bool)
        mask[:inp.shape[0], :inp.shape[1]] = True

    return padded, mask


def read_images(path, max_resize, df=None, padding=True, augment_fn=None, aug_prob=0.0, flip_prob=1.0,
                is_left=None, upper_cornor=None, read_size=None, image=None):
    """
    Args:
        path: string
        max_resize (int): max image size after resied
        df (int, optional): image size division factor.
                            NOTE: this will change the final image size after img_resize
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
        aug_prob (float, optional): probability of applying augment_fn
        flip_prob (float, optional): probability of flipping images
        is_left (bool, optional): if set to 'True', it is left image, otherwise is right image
        upper_cornor (tuple, optional): upper left corner of the image
        read_size (int, optional): read image size
        image (callable, optional): input image
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    assert max_resize is not None
    assert isinstance(max_resize, list)
    if len(max_resize) == 1: max_resize = max_resize * 2

    w_new, h_new = get_divisible_wh(max_resize[0], max_resize[1], df)
    max_resize = [h_new, w_new]

    image = imread_color(path, augment_fn, read_size, image)  # (h,w,3) image is RGB

    # resize image
    w, h = image.shape[1], image.shape[0]
    if (h > max_resize[0]) or (w > max_resize[1]):
        w_new, h_new = get_resized_wh(w, h, max_resize, aug_prob)  # make max(w, h) to max_size
    else:
        w_new, h_new = w, h

    # random resize
    if random.uniform(0, 1) > aug_prob:
        # random rescale
        ratio = max(h / max_resize[0], w / max_resize[1])
        if type(is_left) == bool:
            if is_left:
                low, upper = (0.6 / ratio, 1.0 / ratio) if ratio < 1.0 else (0.6, 1.0)
            else:
                low, upper = (1.0 / ratio, 1.4 / ratio) if ratio < 1.0 else (1.0, 1.4)
        else:
            low, upper = (0.6 / ratio, 1.4 / ratio) if ratio < 1.0 else (0.6, 1.4)
        if not is_left and upper_cornor is not None:
            corner = upper_cornor[2:]
            upper = min(upper, min(max_resize[0]/corner[1], max_resize[1]/corner[0]))
        rands = random.uniform(low, upper)
        w_new, h_new = map(lambda x: x*rands, [w_new, h_new])
        w_new, h_new = get_divisible_wh(w_new, h_new, df)  # make image divided by df and must <= max_size
    else:
        rands = 1
        w_new, h_new = get_divisible_wh(w_new, h_new, df)
        # width, height = w_new, h_new
        # h_start = w_start = 0

    if upper_cornor is not None:
        upper_cornor = upper_cornor[:2]

    # random crop
    if h_new > max_resize[0]:
        height = max_resize[0]
        h_start = int(random.uniform(0, 1) * (h_new - max_resize[0]))
        if upper_cornor is not None:
            h_start = min(h_start, math.floor(upper_cornor[1]*(h_new/h)))
    else:
        height = h_new
        h_start = 0

    if w_new > max_resize[1]:
        width = max_resize[1]
        w_start = int(random.uniform(0, 1) * (w_new - max_resize[1]))
        if upper_cornor is not None:
            w_start = min(w_start, math.floor(upper_cornor[0]*(w_new/w)))
    else:
        width = w_new
        w_start = 0

    w_new, h_new = map(int, [w_new, h_new])
    width, height = map(int, [width, height])

    image = cv2.resize(image, (w_new, h_new))  # (w',h',3)
    image = image[h_start:h_start+height, w_start:w_start+width]

    scale = [w / w_new, h / h_new]
    offset = [w_start, h_start]

    # vertical flip
    if random.uniform(0, 1) > flip_prob:
        hflip = F.hflip_cv2 if image.ndim == 3 and image.shape[2] > 1 and image.dtype == np.uint8 else F.hflip
        image = hflip(image)
        image = F.vflip(image)
        hflip = True
        vflip = True
    else:
        hflip = False
        vflip = False

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # padding
    mask = None
    if padding:
        image, _ = pad_bottom_right(image, max_resize, ret_mask=False)
        gray, mask = pad_bottom_right(gray, max_resize, ret_mask=True)
        mask = torch.from_numpy(mask)

    gray = torch.from_numpy(gray).float()[None] / 255  # (1,h,w)
    image = torch.from_numpy(image).float() / 255  # (h,w,3)
    image = image.permute(2, 0, 1)  # (3,h,w)

    offset = torch.tensor(offset, dtype=torch.float)
    scale = torch.tensor(scale, dtype=torch.float)
    resize = [height, width]

    return gray, image, scale, rands, offset, hflip, vflip, resize, mask
