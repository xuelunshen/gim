import cv2
import torch
import numpy as np
from collections import OrderedDict
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1, offset0, offset1, height0, width0, height1, width1, hfs, vfs):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        pts1 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
        K0:
        K1:
        offset0:
        offset1:
        height0:
        width0:
        height1:
        width1:
        hfs: bool
        vfs: bool
    """
    pts0, pts1 = pts0.clone(), pts1.clone()
    pts0[:, 1] = -pts0[:, 1]+(height0 - 1) if vfs[0] else pts0[:, 1]
    pts0[:, 0] = -pts0[:, 0]+(width0 - 1) if hfs[0] else pts0[:, 0]
    pts1[:, 1] = -pts1[:, 1]+(height1 - 1) if vfs[1] else pts1[:, 1]
    pts1[:, 0] = -pts1[:, 0]+(width1 - 1) if hfs[1] else pts1[:, 0]
    pts0, pts1 = pts0 + offset0[None], pts1 + offset1[None]
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


@torch.no_grad()
def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    offset0 = data['offset0'] * data['scale0']
    offset1 = data['offset1'] * data['scale1']
    width0 = data['resize0'][:,1] * data['scale0'][:, 0]
    width1 = data['resize1'][:,1] * data['scale1'][:, 0]
    hf0 = data['hflip0']
    hf1 = data['hflip1']
    height0 = data['resize0'][:, 0] * data['scale0'][:, 1]
    height1 = data['resize1'][:, 0] * data['scale1'][:, 1]
    vf0 = data['vflip0']
    vf1 = data['vflip1']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(symmetric_epipolar_distance(
            pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs],
            offset0[bs], offset1[bs], height0[bs], width0[bs], height1[bs], width1[bs], (hf0[bs], hf1[bs]), (vf0[bs], vf1[bs])
        ))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def solveEssentialMat(kpts0, kpts1, m):
    kpts0, kpts1, m = map(torch.from_numpy, [kpts0, kpts1, m])

    xx = torch.cat([kpts0, kpts1], dim=1).transpose(0, 1)  # (4, n)
    X = torch.stack([
        xx[2] * xx[0], xx[2] * xx[1], xx[2],
        xx[3] * xx[0], xx[3] * xx[1], xx[3],
        xx[0], xx[1], torch.ones_like(xx[0])
    ], dim=0)  # (9, n)
    wX = m[None] * X  # (9, n)
    XwX = torch.matmul(X, wX.transpose(0, 1))  # (9, 9)

    e_hat = torch.linalg.eigh(XwX, UPLO='U')[1][:, 0]
    e_hat = e_hat / torch.norm(e_hat)
    E = e_hat.reshape((3, 3))

    return E


def estimate_pose(kpts0, kpts1, K0, K1, offset0, offset1, height0, width0, height1, width1, hfs, vfs, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    kpts0[:, 1] = -kpts0[:, 1]+(height0 - 1) if vfs[0] else kpts0[:, 1]
    kpts0[:, 0] = -kpts0[:, 0]+(width0 - 1) if hfs[0] else kpts0[:, 0]
    kpts1[:, 1] = -kpts1[:, 1]+(height1 - 1) if vfs[1] else kpts1[:, 1]
    kpts1[:, 0] = -kpts1[:, 0]+(width1 - 1) if hfs[1] else kpts1[:, 0]
    kpts0, kpts1 = kpts0 + offset0[None], kpts1 + offset1[None]
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        # print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


@torch.no_grad()
def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()
    offset0 = (data['offset0'] * data['scale0']).cpu().numpy()
    offset1 = (data['offset1'] * data['scale1']).cpu().numpy()
    width0 = (data['resize0'][:,1] * data['scale0'][:, 0]).cpu().numpy()
    width1 = (data['resize1'][:,1] * data['scale1'][:, 0]).cpu().numpy()
    hf0 = data['hflip0']
    hf1 = data['hflip1']
    height0 = (data['resize0'][:,0] * data['scale0'][:, 1]).cpu().numpy()
    height1 = (data['resize1'][:,0] * data['scale1'][:, 1]).cpu().numpy()
    vf0 = data['vflip0']
    vf1 = data['vflip1']

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(
            pts0[mask], pts1[mask], K0[bs], K1[bs], offset0[bs], offset1[bs],
            height0[bs], width0[bs], height1[bs], width1[bs], (hf0[bs], hf1[bs]), (vf0[bs], vf1[bs]), pixel_thr, conf=conf
        )

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            data['R_errs'].append(R_err)
            data['t_errs'].append(t_err)
            data['inliers'].append(inliers)


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'AUC@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'Prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4, test=False):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, i) for i, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    metric = {**aucs, **precs}
    metric = {**metric, **{'Num': len(unq_ids)}} if test else metric
    return metric
