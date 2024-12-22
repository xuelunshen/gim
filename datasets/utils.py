# -*- coding: utf-8 -*-
# @Author  : xuelun
import math

import cv2
import torch
import random
import numpy as np

from albumentations.augmentations import functional as F


# ------------
# DATA TOOLS
# ------------
def imread_gray(path, augment_fn=None):
    if augment_fn is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


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


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new = max((w // df), 1) * df
        h_new = max((h // df), 1) * df
        # resize = int(max(max(w, h) // df, 1) * df)
        # w_new, h_new = get_resized_wh(w, h, resize)
        # scale = resize / x
        # w_new, h_new = map(lambda x: int(max(x // df, 1) * df), [w, h])
    else:
        w_new, h_new = w, h
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


def split(n, k):
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)


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

    image = imread_color(path, augment_fn, read_size, image) # (h,w,3) image is RGB

    # resize image
    w, h = image.shape[1], image.shape[0]
    if (h > max_resize[0]) or (w > max_resize[1]):
        w_new, h_new = get_resized_wh(w, h, max_resize, aug_prob) # make max(w, h) to max_size
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
        w_new, h_new = get_divisible_wh(w_new, h_new, df) # make image divided by df and must <= max_size
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

    gray = torch.from_numpy(gray).float()[None] / 255 # (1,h,w)
    image = torch.from_numpy(image).float() / 255  # (h,w,3)
    image = image.permute(2,0,1) # (3,h,w)

    offset = torch.tensor(offset, dtype=torch.float)
    scale = torch.tensor(scale, dtype=torch.float)
    resize = [height, width]

    return gray, image, scale, rands, offset, hflip, vflip, resize, mask


def normalize(pts):
    """Normalizes the input points to [-1, 1]^2 and transforms them to homogenous coordinates.

    Args:
        pts (tensor): input points

    Returns:
        tensor: transformed points
        tensor: transformation
    """

    dim2 = True if pts.size(2) == 2 else False
    factor = 1.0 if dim2 else 1.4142
    pts = torch.cat((pts, torch.ones((pts.size(0), pts.size(1), 1), device=pts.device)), 2) if dim2 else pts

    center = torch.mean(pts, 1)
    dist = pts - center.unsqueeze(1)
    meandist = dist[:, :, :2].pow(2).sum(2).sqrt().mean(1)

    scale = factor / meandist

    transform = torch.zeros((pts.size(0), 3, 3), device=pts.device)

    transform[:, 0, 0] = scale
    transform[:, 1, 1] = scale
    transform[:, 2, 2] = 1
    transform[:, 0, 2] = -center[:, 0] * scale
    transform[:, 1, 2] = -center[:, 1] * scale

    pts_out = torch.bmm(transform, pts.permute(0, 2, 1))

    if dim2:
        return pts_out
    else:
        return pts_out, transform


def weighted_svd(pts1, pts2):
    """Solve homogeneous least squares problem and extract model.

    Args:
        pts1 (tensor): (1, N, 3) points in first image
        pts2 (tensor): (1, N, 3) points in second image

    Returns:
        tensor: estimated fundamental matrix
    """

    pts1n, transform1 = normalize(pts1)  # (1, 3, N), (1, 3, 3)
    pts2n, transform2 = normalize(pts2)  # (1, 3, N), (1, 3, 3)

    X = torch.cat(
        (pts1n[:, 0].unsqueeze(1) * pts2n, pts1n[:, 1].unsqueeze(1) * pts2n, pts2n),
        1,
    ).permute(0, 2, 1)  # (1, N, 9)

    out_batch = []

    mask = torch.tensor([1, 1, 0], dtype=torch.float)
    for batch in range(X.size(0)):
        # solve homogeneous least squares problem
        _, _, V = torch.svd(X[batch])
        F = V[:, -1].view(3, 3)

        # model extractor
        U, S, V = torch.svd(F)
        F_projected = U.mm((S * mask).diag()).mm(V.t())

        out_batch.append(F_projected.unsqueeze(0))

    out = torch.cat(out_batch, 0)  # (1, 3, 3)
    out = transform1.permute(0, 2, 1).bmm(out).bmm(transform2)  # (1, 3, 3)

    return out


def symmetric_epipolar_distance(pts1, pts2, fundamental_mat):
    """Symmetric epipolar distance.

    Args:
        pts1 (tensor): points in first image
        pts2 (tensor): point in second image
        fundamental_mat (tensor): fundamental matrix

    Returns:
        tensor: symmetric epipolar distance
    """

    line_1 = torch.bmm(pts1, fundamental_mat)
    line_2 = torch.bmm(pts2, fundamental_mat.permute(0, 2, 1))

    scalar_product = (pts2 * line_1).sum(2)

    ret = scalar_product.abs() * (
        1 / line_1[:, :, :2].norm(2, 2) + 1 / line_2[:, :, :2].norm(2, 2)
    )

    return ret  # (1, N)
