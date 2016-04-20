import numpy as np
from chainer.cuda import cupy as cp
from utils import cupyutils

xp = None


def generate_inside_anchors(width, height, feat_stride=16, allowed_offset=None,
                            gpu=True):
    """Return a set of anchors for a given image dimension.
    For performance improvement, anchors should be generated once and be
    reused. However, it assumes that the dimensions are the same during the
    whole training and the testing process.
    """
    # TODO: Allow anchors to be slighly outside the image and still be included
    global xp
    xp = cp if gpu else np

    anchors = generate_anchors(gpu=gpu)

    # Apply the 9 anchors to all positions on the filter map
    feat_map_width = width / feat_stride
    feat_map_height = height / feat_stride
    shift_x = xp.arange(0, feat_map_width) * feat_stride
    shift_y = xp.arange(0, feat_map_height) * feat_stride

    if gpu:
        shift_x, shift_y = cupyutils.meshgrid(shift_x, shift_y)
    else:
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = xp.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                       shift_y.ravel())).transpose()

    A = len(anchors)  # 9
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4))
                   .transpose((1, 0, 2)))
    shifts = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 4))

    anchors_inside = []
    for a in all_anchors:
        if _anchor_inside(a, width, height):
            anchors_inside.append(a)

    return anchors_inside


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32],
                     gpu=True):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window."""
    global xp
    xp = cp if gpu else np

    ratios = xp.array(ratios)
    scales = xp.array(scales)
    base_anchor = xp.array([0, 0, base_size - 1, base_size - 1])

    ratio_anchors = _ratio_enum(base_anchor, ratios)

    anchors = xp.vstack([_scale_enum(ratio_anchors[i, :], scales)
                        for i in range(ratio_anchors.shape[0])])

    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, xp.newaxis]
    hs = hs[:, xp.newaxis]
    anchors = xp.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = xp.ceil(xp.sqrt(size_ratios))
    hs = xp.ceil(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _anchor_inside(anchor, img_width, img_height):
    """Return True if the given anchor is completely inside the given image.
    """
    return ((anchor[0] >= 0) & (anchor[1] >= 0) & (anchor[2] < img_width) &
            (anchor[3] < img_height))
