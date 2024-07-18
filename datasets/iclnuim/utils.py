# -*- coding: utf-8 -*-
# @Author  : xuelun
import numpy as np
from imageio import imread


def read_depth(filename):
    depth = np.array(imread(filename))
    depth = depth.astype(np.float32) / 1000
    return depth
