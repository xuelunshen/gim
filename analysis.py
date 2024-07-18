# -*- coding: utf-8 -*-
# @Author  : xuelun

import os
import argparse

import numpy as np

from os.path import join
from datetime import datetime

angular_thresholds = ['5.0°']
dist_thresholds = ['0.1m']
intt = lambda x: list(map(int, x))
floatt = lambda x: list(map(float, x))
strr = lambda x: list(map(lambda x:f'{x:.18f}', x))

datasets = [
    'GL3D',
    'BlendedMVS',
    'ETH3DI',
    'ETH3DO',
    'KITTI',
    'RobotcarWeather',
    'RobotcarSeason',
    'RobotcarNight',
    'Multi-FoV',
    'SceneNetRGBD',
    'ICL-NUIM',
    'GTA-SfM',
]


def error_auc(errs0, errs1, thres, metric):
    if isinstance(errs0, list): errs0 = np.array(errs0)
    if isinstance(errs1, list): errs1 = np.array(errs1)
    if any(np.isnan(errs0)): errs0[np.isnan(errs0)] = 180
    if any(np.isnan(errs1)): errs1[np.isnan(errs1)] = 180
    if any(np.isinf(errs0)): errs0[np.isinf(errs0)] = 180
    if any(np.isinf(errs1)): errs1[np.isinf(errs1)] = 180
    errors = np.max(np.stack([errs0, errs1]), axis=0)
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thres:
        thr = float(thr[:-1])
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'{metric}@ {t}': auc for t, auc in zip(thres, aucs)}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.')
    parser.add_argument('--wid', type=str, required=True)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--sceids', type=str, choices=datasets, nargs='+',
                        default=None, help=f'Test Datasets: {datasets}', )
    opt = parser.parse_args()

    dir = opt.dir
    wid = opt.wid
    version = opt.version

    _data = \
        {
            x.rpartition('.txt')[0].split()[2]:x for x in
            [
                d for d in os.listdir(dir) if not os.path.isdir(os.path.join(dir, d))
            ] if wid == x.rpartition('.txt')[0].split()[1] and version is not None and version == x.rpartition('.txt')[0].split()[-1]
        }
    _data = {k:_data[k] for k in datasets if k in _data.keys()}

    sceids = opt.sceids
    sceids = sceids if sceids is not None else _data.keys()
    results = {}
    for sceid in sceids:
        results[sceid] = {}
        if not opt.verbose: print('{:^13} {}'.format(sceid, wid))

        # read txt
        with open(join(dir, _data[sceid]), 'r') as f:
            data = f.readlines()
        head = data[0].split()
        content = [x.split() for x in data[1:]]
        details = {k: [] for k in head[3:]}

        stacks = []
        for x in content:
            ids = x[0]
            if ids in stacks: continue

            for k, v in zip(head[3:], x[3:]): details[k].append(v)
            stacks.append(ids)

        mAP = error_auc(floatt(details['R_errs']), floatt(details['t_errs']), angular_thresholds, 'auc')
        for k, v in mAP.items(): results[sceid][k] = v

    # print head
    output = ''

    num = 56+25*len(sceids)
    output += '='*num
    output += "\n"

    output += '{:<25}'.format(datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))
    output += '{:<15} '.format('Model')
    output += '{:<14} '.format('Metric')
    for sceid in sceids: output += '{:<25} '.format(sceid)
    output += "\n"

    output += '-'*num
    output += "\n"

    for k in list(results.values())[0].keys():
        output += '{:<25}'.format(datetime.now().strftime("%Y-%m-%d, %H:%M:%S")) if opt.log else '{:<25}'.format(' ')
        output += '{:<15} '.format(wid)
        output += '{:<14} '.format(k)

        for sceid in sceids:
            output += '{:<25} '.format(results[sceid][k])
        output += "\n"

    output += '='*num
    output += "\n"
    output += "\n"

    if opt.verbose:
        print(output)

    if opt.log:
        path = 'ANALYSIS RESULTS.txt'
        with open(path, 'a') as file:
            file.write(output)
