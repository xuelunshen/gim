# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import time
import yaml
import torch
import random
import numpy as np


project_name = os.path.basename(os.getcwd())


def make_reproducible(iscuda, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if iscuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # set True will make data load faster
        #   but, it will influence reproducible
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def hint(msg):
    timestamp = f'{time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))}'
    print('\033[1m' + project_name + ' >> ' + timestamp + ' >> ' + '\033[0m' + msg)


def datainfo(infos, datalen, gpuid):
    if gpuid != 0: return
    # print informations about benchmarks
    print('')
    print(f'{" Benchmarks":14}|{" Sequence":20}|{" Count":8}')
    print(f'{"-" * 45}')
    for k0, v0 in infos.items():
        isfirst = True
        for k1, v1 in v0.items():
            line = f' {k0:13}|' if isfirst else f'{" " * 14}|'
            line += f' {k1:19}|'
            line += f' {str(v1):7}'
            print(line)
            print(f'{"-" * 45}')
            isfirst = False
    print(f'{" " * 37}{str(datalen)}')
    print(f'{"-" * 45}')
    print('')


# noinspection PyTypeChecker
def mesh_positions(h: int, w: int):
    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w))
    gx, gy = gx.contiguous()[None, :], gy.contiguous()[None, :]
    pos = torch.cat((gx.view(1, -1), gy.view(1, -1)))  # [2, H*W]
    return pos


def current_time(f=None):
    """
    :param f: default for log, "f" for file name
    :return: formatted time
    """
    if f == "f":
        return f'{time.strftime("%m.%d_%H.%M.%S", time.localtime(time.time()))}'
    return f'{time.strftime("%m/%d %H:%M:%S", time.localtime(time.time()))}'


def mkdir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=False)


def pdist(x, y=None):
    """
    Pairwise Distance
    Args:
        x: [bs, n, 2]
        y: [bs, n, 2]
    Returns: [bs, n, n] value in euclidean *square* distance
    """
    # B, n, two = x.shape
    x = x.double()  # [bs, n, 2]
    
    x_norm = (x ** 2).sum(-1, keepdim=True)  # [bs, n, 1]
    if y is not None:
        y = y.double()
        y_t = y.transpose(1, 2)  # [bs, 2, n]
        y_norm = (y ** 2).sum(-1, keepdim=True).transpose(1, 2)  # [bs, 1, n]
    else:
        y_t = x.transpose(1, 2)  # [bs, 2, n]
        y_norm = x_norm.transpose(1, 2)  # [bs, 1, n]
    
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, y_t)  # [bs, n, n]
    return dist


mean = lambda lis: sum(lis) / len(lis)
eps = lambda x: x + 1e-8


def load_configs(configs):
    with open(configs, 'r') as stream:
        try:
            x = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return x


def find_in_dir(run, dir):
    runs = os.listdir(dir)
    runs = [r for r in runs if run in r]
    if len(runs) <= 0:
        hint(f'Not exist run name contain : {run}')
        exit(-1)
    elif len(runs) >= 2:
        hint(f'{len(runs)} runs name contain : {run}')
        hint(f'I will return the first one : {runs[-1]}')
    else:
        hint(f'Success match {runs[-1]}')
    return runs[-1]


def ckpt_in_dir(key, dir):
    runs = os.listdir(dir)
    runs = [r for r in runs if key in r]
    if len(runs) <= 0:
        hint(f'Not exist run name contain : {key}')
        exit(-1)
    elif len(runs) >= 2:
        hint(f'{len(runs)} runs name contain : {key}')
        hint(f'I will return the first one : {runs[-1]}')
    else:
        hint(f'Success match {runs[-1]}')
    return runs[-1]


def kpts2grid(kpts, scale, size):
    """
    change coordinates for keypoints from size0 to size1
    and format as grid which coordinates from [-1, 1]
    Args:
        kpts: (b, n, 2) - (x, y)
        scale: (b, 2) - (w, h) - the keypoints working shape to unet working shape 
        size: (b, 2) - (h, w) - the unet working shape which is 'resize0/1' in data
    Returns: new kpts: (b, 1, n, 2) - (x, y) in [-1, 1]
    """
    # kpts coordinates in unet shape
    kpts /= scale[:,None,:]
    # kpts[:,:,0] - (b, n)
    kpts[:, :, 0] *= 2 / (size[:, 1][:, None] - 1)
    kpts[:, :, 1] *= 2 / (size[:, 0][:, None] - 1)
    # make kpts from [0, 2] to [-1, 1]
    kpts -= 1
    # assume all kpts in [-1, 1]
    kpts = kpts.clamp(min=-1, max=1) # (b, n, 2)
    # make kpts shape from (b, n, 2) to (b, 1, n, 2)
    kpts = kpts[:,None]

    return kpts


def debug(x):
    if 'DATASET' in list(x.keys()):
        y = x.DATASET
        y.TRAIN.LIST_PATH = y.TRAIN.LIST_PATH.replace('scene_list', 'scene_list_debug')
        y.VALID.LIST_PATH = y.VALID.LIST_PATH.replace('scene_list', 'scene_list_debug')
    return x


def summary_loss(loss_list):
    n = 0
    sums = 0
    for loss in loss_list:
        if (loss is not None) and (not torch.isnan(loss)):
            sums += loss
            n += 1
    sums = sums / n if n != 0 else None
    return sums


def summary_metrics(dic, h1, h2):
    print('')

    # Head
    print(f'RunID     {h1:9}', end='')
    print('   | ', end='')
    print(f'Version   {h2:10}', end='')

    # Content
    print(f'{"| ".join(f"{key:10}" for key in dic[0].keys())}')
    for metric in dic:
        print(f'{"-" * 12 * len(dic[0].keys())}')
        print(f'{"| ".join(f"{metric[key]:<10.5f}" for key in metric.keys())}')

    print('')


def get_padding_size(image, h, w):
    orig_width = image.shape[3]
    orig_height = image.shape[2]
    aspect_ratio = w / h

    new_width = max(orig_width, int(orig_height * aspect_ratio))
    new_height = max(orig_height, int(orig_width / aspect_ratio))

    pad_height = new_height - orig_height
    pad_width = new_width - orig_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return orig_width, orig_height, pad_left, pad_right, pad_top, pad_bottom
