# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import torch
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from os.path import join
from tools import get_padding_size
from networks.loftr.loftr import LoFTR
from networks.loftr.misc import lower_config
from networks.loftr.config import get_cfg_defaults
from networks.dkm.models.model_zoo.DKMv3 import DKMv3
from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue

DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_METHOD = "USAC_MAGSAC"

RANSAC_ZOO = {
    "RANSAC": cv2.RANSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    # elif interp.startswith('pil_'):
    #     interp = getattr(PIL.Image, interp[len('pil_'):].upper())
    #     resized = PIL.Image.fromarray(image.astype(np.uint8))
    #     resized = resized.resize(size, resample=interp)
    #     resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def fast_make_matching_figure(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().detach().numpy()
    kpts1 = data['mkpts1_f'].cpu().detach().numpy()
    mconf = data['mconf'].cpu().detach().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def fast_make_matching_overlay(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().detach().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().detach().numpy()
    kpts1 = data['mkpts1_f'].cpu().detach().numpy()
    mconf = data['mconf'].cpu().detach().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.line(out, (x0, y0 + sh), (x1 + margin + w0, y1 + sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(
            lambda x: int(x // dfactor * dfactor),
            image.shape[-2:]))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale


def compute_geom(data,
                 ransac_method=DEFAULT_RANSAC_METHOD,
                 ransac_reproj_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
                 ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
                 ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
                 ) -> dict:

    mkpts0 = data["mkpts0_f"].cpu().detach().numpy()
    mkpts1 = data["mkpts1_f"].cpu().detach().numpy()

    if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
        return {}

    h1, w1 = data["hw0_i"]

    geo_info = {}

    F, inliers = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if F is not None:
        geo_info["Fundamental"] = F.tolist()

    H, _ = cv2.findHomography(
        mkpts1,
        mkpts0,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if H is not None:
        geo_info["Homography"] = H.tolist()
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            mkpts0.reshape(-1, 2),
            mkpts1.reshape(-1, 2),
            F,
            imgSize=(w1, h1),
        )
        geo_info["H1"] = H1.tolist()
        geo_info["H2"] = H2.tolist()

    return geo_info


def wrap_images(img0, img1, geo_info, geom_type):
    img0 = img0[0].permute((1, 2, 0)).cpu().detach().numpy()[..., ::-1]
    img1 = img1[0].permute((1, 2, 0)).cpu().detach().numpy()[..., ::-1]

    h1, w1, _ = img0.shape
    h2, w2, _ = img1.shape

    rectified_image0 = img0
    rectified_image1 = None
    H = np.array(geo_info["Homography"])
    F = np.array(geo_info["Fundamental"])

    title = []
    if geom_type == "Homography":
        rectified_image1 = cv2.warpPerspective(
            img1, H, (img0.shape[1], img0.shape[0])
        )
        title = ["Image 0", "Image 1 - warped"]
    elif geom_type == "Fundamental":
        H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
        rectified_image0 = cv2.warpPerspective(img0, H1, (w1, h1))
        rectified_image1 = cv2.warpPerspective(img1, H2, (w2, h2))
        title = ["Image 0 - warped", "Image 1 - warped"]
    else:
        print("Error: Unknown geometry type")

    fig = plot_images(
        [rectified_image0.squeeze(), rectified_image1.squeeze()],
        title,
        dpi=300,
    )

    img = fig2im(fig)

    plt.close(fig)

    return img


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=5, pad=0.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        dpi:
        size:
        pad:
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    figsize = (size * n, size * 6 / 5) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    fig.tight_layout(pad=pad)

    return fig


def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.buffer_rgba(), dtype="u1")
    # noinspection PyArgumentList
    im = buf_ndarray.reshape(h, w, 4)
    return im


if __name__ == '__main__':
    model_zoo = ['gim_dkm', 'gim_loftr', 'gim_lightglue']

    # model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gim_dkm', choices=model_zoo)
    args = parser.parse_args()

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    ckpt = None
    model = None
    detector = None
    if args.model == 'gim_dkm':
        ckpt = 'gim_dkm_100h.ckpt'
        model = DKMv3(weights=None, h=672, w=896)
    elif args.model == 'gim_loftr':
        ckpt = 'gim_loftr_50h.ckpt'
        model = LoFTR(lower_config(get_cfg_defaults())['loftr'])
    elif args.model == 'gim_lightglue':
        ckpt = 'gim_lightglue_100h.ckpt'
        detector = SuperPoint({
            'max_num_keypoints': 2048,
            'force_num_keypoints': True,
            'detection_threshold': 0.0,
            'nms_radius': 3,
            'trainable': False,
        })
        model = LightGlue({
            'filter_threshold': 0.1,
            'flash': False,
            'checkpointed': True,
        })

    # weights path
    checkpoints_path = join('weights', ckpt)

    # load state dict
    if args.model == 'gim_dkm':
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)

    elif args.model == 'gim_loftr':
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

    elif args.model == 'gim_lightglue':
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        detector.load_state_dict(state_dict)

        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    # eval mode
    if detector is not None:
        detector = detector.eval().to(device)
    model = model.eval().to(device)

    name0 = 'a1'
    name1 = 'a2'
    postfix = '.png'
    image_dir = join('assets', 'demo')
    img_path0 = join(image_dir, name0 + postfix)
    img_path1 = join(image_dir, name1 + postfix)

    image0 = read_image(img_path0)
    image1 = read_image(img_path1)
    image0, scale0 = preprocess(image0)
    image1, scale1 = preprocess(image1)

    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]

    b_ids, mconf, kpts0, kpts1 = None, None, None, None
    data = dict(color0=image0, color1=image1, image0=image0, image1=image1)

    if args.model == 'gim_dkm':
        orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = get_padding_size(image0, 672, 896)
        orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = get_padding_size(image1, 672, 896)
        image0_ = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
        image1_ = torch.nn.functional.pad(image1, (pad_left1, pad_right1, pad_top1, pad_bottom1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense_matches, dense_certainty = model.match(image0_, image1_)
            sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 5000)

        height0, width0 = image0_.shape[-2:]
        height1, width1 = image1_.shape[-2:]

        kpts0 = sparse_matches[:, :2]
        kpts0 = torch.stack((
            width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
        kpts1 = sparse_matches[:, 2:]
        kpts1 = torch.stack((
            width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)
        b_ids = torch.where(mconf[None])[0]

        # before padding
        kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
        kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
        mask_ = (kpts0[:, 0] > 0) & \
               (kpts0[:, 1] > 0) & \
               (kpts1[:, 0] > 0) & \
               (kpts1[:, 1] > 0)
        mask_ = mask_ & \
               (kpts0[:, 0] <= (orig_width0 - 1)) & \
               (kpts1[:, 0] <= (orig_width1 - 1)) & \
               (kpts0[:, 1] <= (orig_height0 - 1)) & \
               (kpts1[:, 1] <= (orig_height1 - 1))

        mconf = mconf[mask_]
        b_ids = b_ids[mask_]
        kpts0 = kpts0[mask_]
        kpts1 = kpts1[mask_]

    elif args.model == 'gim_loftr':
        with torch.no_grad():
            model(data)
        kpts0 = data['mkpts0_f']
        kpts1 = data['mkpts1_f']
        b_ids = data['m_bids']
        mconf = data['mconf']

    elif args.model == 'gim_lightglue':
        gray0 = read_image(img_path0, grayscale=True)
        gray1 = read_image(img_path1, grayscale=True)
        gray0 = preprocess(gray0, grayscale=True)[0]
        gray1 = preprocess(gray1, grayscale=True)[0]

        gray0 = gray0.to(device)[None]
        gray1 = gray1.to(device)[None]
        scale0 = torch.tensor(scale0).to(device)[None]
        scale1 = torch.tensor(scale1).to(device)[None]

        data.update(dict(gray0=gray0, gray1=gray1))

        size0 = torch.tensor(data["gray0"].shape[-2:][::-1])[None]
        size1 = torch.tensor(data["gray1"].shape[-2:][::-1])[None]

        data.update(dict(size0=size0, size1=size1))
        data.update(dict(scale0=scale0, scale1=scale1))

        pred = {}
        with torch.no_grad():
            pred.update({k + '0': v for k, v in detector({
                "image": data["gray0"],
            }).items()})
            pred.update({k + '1': v for k, v in detector({
                "image": data["gray1"],
            }).items()})
            pred.update(model({**pred, **data,
                               **{'image_size0': data['size0'],
                                  'image_size1': data['size1']}}))

        kpts0 = torch.cat([kp * s for kp, s in zip(pred['keypoints0'], data['scale0'][:, None])])
        kpts1 = torch.cat([kp * s for kp, s in zip(pred['keypoints1'], data['scale1'][:, None])])
        m_bids = torch.nonzero(pred['keypoints0'].sum(dim=2) > -1)[:, 0]
        matches = pred['matches']
        bs = data['image0'].size(0)
        kpts0 = torch.cat([kpts0[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        kpts1 = torch.cat([kpts1[m_bids == b_id][matches[b_id][..., 1]] for b_id in range(bs)])
        b_ids = torch.cat([m_bids[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
        mconf = torch.cat(pred['scores'])

    # robust fitting
    _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
                                     kpts1.cpu().detach().numpy(),
                                     cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                     confidence=0.999999, maxIters=10000)
    mask = mask.ravel() > 0

    data.update({
        'hw0_i': image0.shape[-2:],
        'hw1_i': image1.shape[-2:],
        'mkpts0_f': kpts0,
        'mkpts1_f': kpts1,
        'm_bids': b_ids,
        'mconf': mconf,
        'inliers': mask,
    })

    # save visualization
    alpha = 0.5
    out = fast_make_matching_figure(data, b_id=0)
    overlay = fast_make_matching_overlay(data, b_id=0)
    out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(join(image_dir, f'{name0}_{name1}_{args.model}_match.png'), out[..., ::-1])

    geom_info = compute_geom(data)
    wrapped_images = wrap_images(image0, image1, geom_info,
                                 "Homography")
    cv2.imwrite(join(image_dir, f'{name0}_{name1}_{args.model}_warp.png'), wrapped_images)
