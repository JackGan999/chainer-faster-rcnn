import numpy as np
from chainer.cuda import cupy as cp

# TODO Hardcoded to use GPU
xp = cp


def ious_slow(anchors, gt_boxes):
    """Return a set IOU(Intersection-Over-Union)s for ground-truth box for
    each anchor. Naive implementation.
    """
    return [iou_naive(anchor, gt_box) for anchor in anchors
            for gt_box in gt_boxes]


def iou_naive(anchor, gt_box):
    """Return the IOU for the given anchor ground-truth box pair.
    Naive/Slow.
    """
    anchor_x1 = anchor[0]
    anchor_y1 = anchor[1]
    anchor_x2 = anchor[2]
    anchor_y2 = anchor[3]

    gt_box_x1 = gt_box[0]
    gt_box_y1 = gt_box[1]
    gt_box_x2 = gt_box[2]
    gt_box_y2 = gt_box[3]

    area_intersection = (max(0, min(anchor_x2, gt_box_x2) -
                         max(anchor_x1, gt_box_x1)) *
                         max(0, min(anchor_y2, gt_box_y2) -
                         max(anchor_y1, gt_box_y1)))

    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt_box = (gt_box_x2 - gt_box_x1) * (gt_box_y2 - gt_box_y1)

    area_union = area_anchor + area_gt_box - area_intersection

    return area_intersection / area_union


def ious(boxes, query_boxes):
    """Return a set IOU(Intersection-Over-Union)s for ground-truth box for
    each anchor. Naive implementation.
    """
    # TODO: Improve speed, e.g. test range() instead of enumerate()
    overlaps = xp.zeros((boxes.shape[0], query_boxes.shape[0]),
                        dtype=np.float32)

    for q_i, q in enumerate(query_boxes):
        q_area = (q[2] - q[0] + 1) * (q[3] - q[1] + 1)  # Area of the query box
        for b_i, b in enumerate(boxes):
            iw = min(b[2], q[2]) - max(b[0], q[0]) + 1
            if iw > 0:
                ih = min(b[3], q[3]) - max(b[1], q[1]) + 1
                if ih > 0:
                    ua = ((b[2] - b[0] + 1) * (b[3] - b[1] + 1) +
                          q_area - iw * ih)
                    overlaps[b_i, q_i] = iw * ih / ua

    return overlaps


def iou_gpu_0(anchor, gt_box):
    """Compute the intersection over union rate for the given anchor and a
    gt_box. Not very fast, but works...
    """
    return cp.ElementwiseKernel(
        'raw float32 anchor, raw float32 gt_box',
        'float32 iou',
        '''
            float inters = max(0.0, min(anchor[2], gt_box[2]) -
                max(anchor[0], gt_box[0])) *
                max(0.0, min(anchor[3], gt_box[3]) -
                max(anchor[1], gt_box[1]));
            float anchor_area = (anchor[2] - anchor[0]) *
                (anchor[3] - anchor[1]);
            float gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]);
            float union_area = anchor_area + gt_area - inters;

            iou = inters / union_area;
        ''', 'intersection_over_union'
      )(anchor, gt_box, size=1)  # Is size=1 fine?


def ious_gpu_1(boxes, query_boxes):
    """Kernel function IOU computation."""
    # TODO: Fix, does not work. Not using ElementwiseKernel correct.
    n_boxes = boxes.shape[0]
    n_query_boxes = query_boxes.shape[0]

    print(n_boxes)
    print(n_query_boxes)
    print(boxes)
    print(query_boxes)

    ious = cp.zeros((n_query_boxes, n_boxes), dtype=cp.float32)

    print(ious)

    cp.ElementwiseKernel(
    '''raw float32 boxes, float32 query_boxes, raw int32 num_boxes,
    raw int32 num_query_boxes
    ''',
    'raw float32 ious',
    '''
        for (int q = 0; q < num_query_boxes; ++q) {
            float box_area = (query_boxes[q, 2] - query_boxes[q, 0] + 1.0) *
                (query_boxes[q, 3] - query_boxes[q, 1] + 1.0);
            ious[q, 0] = q;
            for (int b = 0; b < num_boxes; ++b) {
                float iw = min(boxes[b, 2], query_boxes[q, 2]) -
                    max(boxes[b, 0], query_boxes[q, 0]) + 1.0;
                if (iw > 0.0) {
                    float ih = min(boxes[b, 3], query_boxes[q, 3]) -
                        max(boxes[b, 1], query_boxes[q, 1]) + 1.0;
                    if (ih > 0.0) {
                        float ua = (boxes[b, 2] - boxes[b, 0] + 1.0) *
                            (boxes[b, 3] - boxes[b, 1] + 1.0) +
                            box_area - (iw * ih);
                         // ious[q, b] = q;
                      //ious[q, b] = (iw * ih) / ua;
                    }
                } else {
                    ious[q, b] = -1.1;
                }
            }
        }
    ''',
    'intersecion_over_unions'
    )(boxes, query_boxes, n_boxes, n_query_boxes, ious, size=1)
    return ious


def ious_gpu_2(boxes, query_boxes):
    """Kernel function IOU computation."""
    # TODO: Fix, does not work. Not using ElementwiseKernel correct.
    n_boxes = boxes.shape[0]
    n_query_boxes = query_boxes.shape[0]

    print(n_boxes)
    print(n_query_boxes)
    print(boxes)
    print(query_boxes)

    ious = cp.zeros((n_query_boxes, n_boxes), dtype=cp.float32)

    print(ious)

    cp.ElementwiseKernel(
    '''raw float32 boxes, raw float32 query_boxes, raw int32 num_boxes,
    raw int32 num_query_boxes
    ''',
    'raw float32 ious',
    '''
        for (int q = 0; q < num_query_boxes; ++q) {
            float box_area = (query_boxes[q, 2] - query_boxes[q, 0] + 1.0) *
                (query_boxes[q, 3] - query_boxes[q, 1] + 1.0);
            ious[q, 0] = q;
            for (int b = 0; b < num_boxes; ++b) {
                float iw = min(boxes[b, 2], query_boxes[q, 2]) -
                    max(boxes[b, 0], query_boxes[q, 0]) + 1.0;
                if (iw > 0.0) {
                    float ih = min(boxes[b, 3], query_boxes[q, 3]) -
                        max(boxes[b, 1], query_boxes[q, 1]) + 1.0;
                    if (ih > 0.0) {
                        float ua = (boxes[b, 2] - boxes[b, 0] + 1.0) *
                            (boxes[b, 3] - boxes[b, 1] + 1.0) +
                            box_area - (iw * ih);
                         // ious[q, b] = q;
                      //ious[q, b] = (iw * ih) / ua;
                    }
                } else {
                    ious[q, b] = -1.1;
                }
            }
        }
    ''',
    'intersecion_over_unions'
    )(boxes, query_boxes, n_boxes, n_query_boxes, ious, size=1)
    return ious


# def test_kernel(a, b):
#     print('a.shape {}'.format(a.shape))
#     print('b.shape {}'.format(b.shape))
#     print('a {}'.format(a))
#     print('b {}'.format(b))
#     ans = cp.zeros((a.shape[0]* b.shape[0]), dtype=cp.float32)
#     size = a.shape[0] * b.shape[0]
#     print('size {}'.format(size))
#     return cp.ElementwiseKernel(
#     'raw T a, raw T b',
#     'T ans',
#     '''
#         int a_idx = i % n_boxes,
#         ans = i;
#         // ans = a[i, 0, 0] + b[0, i, 0];
#     ''',
#     'testing'
#     )(a, b, ans, size=size)

# def test_kernel(a, b):
#     print('a.shape {}'.format(a.shape))
#     print('b.shape {}'.format(b.shape))
#     print('a {}'.format(a))
#     print('b {}'.format(b))
#     ans = cp.zeros((b.shape[0] * a.shape[0]), dtype=cp.float32)
#     size = a.shape[0] * b.shape[0]
#
#     print('size {}'.format(size))
#     # TODO Seems like a and b are ravel()-ed so index with [i] instead of [i, j]
#     cp.ElementwiseKernel(
#     'raw float32 boxes, raw float32 qboxes, raw int32 n_b, raw int32 n_qb',
#     'T iou',
#     '''
#         int q = i % n_qb;
#         int b = i % n_b;
#
#
#         //float box_area = (qboxes[(q * n_q) + 2] - qboxes[q * n_q] + 1.0) * (qboxes[(q * n_q) + 3] - qboxes[(q * n_q) + 1] + 1.0);
#         float box_area = -1.0;
#         float iw = min(boxes[b, 2], qboxes[q, 2]) - max(boxes[b, 0], qboxes[q, 0]) + 1.0;
#         if (iw > 0.0) {
#             float ih = min(boxes[b, 3], qboxes[q, 3]) - max(boxes[b, 1], qboxes[q, 1]) + 1.0;
#             if (ih > 0.0) {
#                 float ua = (boxes[b, 2] - boxes[b, 0] + 1.0) * (boxes[b, 3] - boxes[b, 1] + 1.0) +
#                     box_area - (iw * ih);
#                 iou = (iw * ih) / ua;
#             }
#         }
#         int ind = q * n_q;
#         iou = qboxes[ind];
#     ''',
#     'testing'
#     )(a, b, a.shape[0], b.shape[0], ans, size=size)
#     return ans
