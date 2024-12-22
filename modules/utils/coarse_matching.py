import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops.einops import rearrange

INF = 1e9


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def zeroshot_coarse_matching(feat_0, feat_1, data, mask_c0=None, mask_c1=None, temperature=None, sample_num=None):

    # reshape feat_c0 to hw0_c
    axes_lengths = {'h0c': data['hw0_c'][0], 'w0c': data['hw0_c'][1]}
    feat_c0 = rearrange(feat_0, 'b (h0c w0c) c -> b c h0c w0c', **axes_lengths)

    # reshape feat_c1 to hw1_c
    axes_lengths = {'h1c': data['hw1_c'][0], 'w1c': data['hw1_c'][1]}
    feat_c1 = rearrange(feat_1, 'b (h1c w1c) c -> b c h1c w1c', **axes_lengths)

    # get pseudo labels
    zs = data['zs']
    znum = zs.sum()
    pseudo_labels = data['pseudo_labels'][zs]
    b_ids, n_ids = torch.where(pseudo_labels.sum(dim=2) > 0)
    pseudo_labels = pseudo_labels[b_ids, n_ids]  # (n, 4)
    pt0 = pseudo_labels[:, :2]  # (n, 2), in hw_i(image) size coordinates
    pt1 = pseudo_labels[:, 2:]  # (n, 2), in hw_i(image) size coordinates

    unique_b = torch.unique(b_ids)
    if (sample_num > 0) and len(b_ids) > (sample_num * len(unique_b)):
        indices = torch.cat([
            torch.randperm((b_ids == b).sum(), device=feat_c0.device)[:sample_num]
            + (b_ids < b).sum()
            for b in unique_b])
        b_ids, pt0, pt1 = b_ids[indices], pt0[indices], pt1[indices]

    grid_sample = partial(F.grid_sample, align_corners=True, mode='bilinear')
    scale = data['hw0_i'][0] / data['hw0_c'][0]

    # sample coarse-level descriptors
    grid0 = pt_to_grid(pt0.clone()[None]/scale, data['hw0_c'])  # (1, 1, n, 2)
    feat_c0 = [grid_sample(feat_c0[[i]], grid0[:, :, b_ids == i]) for i in range(znum)]  # [(1, 256, 1, n)]
    feat_c0 = torch.cat([x.squeeze().transpose(0, 1) for x in feat_c0], dim=0)  # (n, 256)

    # sample coarse-level descriptors
    grid1 = pt_to_grid(pt1.clone()[None]/scale, data['hw1_c'])  # (1, 1, n, 2)
    feat_c1 = [grid_sample(feat_c1[[i]], grid1[:, :, b_ids == i]) for i in range(znum)]  # [(1, 256, 1, n)]
    feat_c1 = torch.cat([x.squeeze().transpose(0, 1) for x in feat_c1], dim=0)  # (n, 256)

    # normalize
    feat0, feat1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_c0, feat_c1])
    feat_0, feat_1 = map(lambda feat: feat / feat.shape[-1] ** .5, [feat_0, feat_1])

    # dual softmax
    b_num = [(b_ids==i).sum().item() for i in range(znum)]
    sim_matrix = [torch.einsum(
        "lc,sc->ls",
        torch.cat((feat0[b_ids==i], feat_0[i]), dim=0),
        torch.cat((feat1[b_ids==i], feat_1[i]), dim=0)
    ) / temperature for i in range(znum)]
    sim_matrix = [
        mat.masked_fill_(~(
                torch.cat((m0.new_ones(n).bool(), m0))[:, None] *
                torch.cat((m1.new_ones(n).bool(), m1))[None]
        ).bool(), -INF) for mat, m0, m1, n in zip(sim_matrix, mask_c0, mask_c1, b_num)
    ]
    conf_matrix = [F.softmax(mat, 0) * F.softmax(mat, 1) for mat in sim_matrix]
    conf_matrix = [mat[:n, :n] for mat, n in zip(conf_matrix, b_num)]

    data.update({
        'zs_pt0_i': pt0,  # (n, 2)
        'zs_pt1_i': pt1,  # (n, 2)
        'zs_b_ids': b_ids,  # (n,)
        'zs_feat_c0': feat_c0,  # (n, 256)
        'zs_feat_c1': feat_c1,  # (n, 256)
        'zs_conf_matrix': conf_matrix,  # [(n', n'), (m', m'), ...]
    })


def pt_to_grid(pt, hw):
    """
    Args:
        pt: (b, n, 2) - (x, y)
        hw: (2) - (h, w) - the kpts working size coordinates

    Returns: grid pt: (b, 1, n, 2) - (x, y) in [-1, 1]
    """
    # make pts to [0, 2]
    pt[:, :, 0] *= 2 / (hw[1] - 1)
    pt[:, :, 1] *= 2 / (hw[0] - 1)
    # make pts from [0, 2] to [-1, 1]
    pt -= 1
    # make sure all pts in [-1, 1]
    assert (pt >= -1).all() and (pt <= 1).all()
    # make pts shape from (b, n, 2) to (b, 1, n, 2)
    pt = pt[:, None]

    return pt


# https://github.com/pytorch/pytorch/issues/36748#issuecomment-1072093200
def unique(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # noinspection PyArgumentList
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat/feat.shape[-1]**.5, [feat_c0, feat_c1])

        conf_matrix = None
        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)/self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config['sparse_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})

        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        gt = data['gt']
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'][gt], data['mask1'][gt])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'][gt], data['mask1'][gt])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.train_pad_num_gt_min \
                if self.train_pad_num_gt_min < num_matches_train \
                else num_matches_train // 2

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']),
                (max(num_matches_train - num_matches_pred, train_pad_num_gt_min), ),
                device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        if len(b_ids) == 0:
            # logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
            # this won't affect fine-level loss calculation
            b_ids = torch.tensor([0], device=_device)
            i_ids = torch.tensor([0], device=_device)
            j_ids = torch.tensor([0], device=_device)

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        # scale = data['hw0_i'][0] / data['hw0_c'][0]
        # scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        # scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
            dim=1)
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
            dim=1)

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
            'mconf': mconf,
        })

        return coarse_matches
