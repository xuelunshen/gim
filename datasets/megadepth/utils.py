# -*- coding: utf-8 -*-
# @Author  : xuelun

import h5py
import numpy as np
from skimage.filters import threshold_otsu as otsu
from datasets.utils import pad_bottom_right


# ------------
# MEGADEPTH
# ------------
def read_megadepth_depth(path, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    # depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


def configure_depth(depth):

    dp = depth.copy()

    # Do not know why use dep_a.sum() will raise
    # /home/sxl/.local/lib/python3.6/site-packages/numpy/core/_methods.py:47:
    # RuntimeWarning: overflow encountered in reduce
    # return umr_sum(a, axis, dtype, out, keepdims, initial, where)
    # hence, replace dep_a.sum() to sum(sum(dep_a))
    depth = depth - np.min(depth[depth >= 1e-8]) if sum(sum(depth)) > 0 else depth

    depth = depth / max(depth.max(), 1e-8)

    depth = (dp >= 1e-8) * depth
    
    return np.abs(depth)


def OTSU(x):
    y = (x*255).copy().astype(np.uint8)
    t = otsu(y[x!=0]) / 255
    mask0 = x >= t
    mask1 = (~mask0) & (x != 0)
    x = x.astype(np.int8)
    x[mask0] = 1
    x[mask1] = -1
    return x
