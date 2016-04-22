import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check

from utils import cupyutils
from utils import iouutils
from utils import imgutils
from utils import anchorutils


def coord_param(box, anchor):
    """
    x, y, w, h
    """
    xp = cuda.get_array_module(box, anchor)
    xy = (box[0:2] - anchor[0:2]) / anchor[2:4]
    wh = xp.log(box[2:4] / anchor[2:4])
    return xp.concatenate([xy, wh])


def loss_cls(p, p_start):
    raise NotImplementedError('loss_cls')


def loss_bbox(t, t_star):
    sum(smooth_l1(t_star - t))


def smooth_l1(xs):
    xp = cuda.get_array_module(xs)
    return [0.5 * x * x if x < 1 else x - 0.5 for x in xp.abs(xs)]


class MultiTask(function.Function):

    def __init__(self, lbd, spatial_scale, use_cudnn=True):
        self.lbd = lbd  # The lambda parameter mentioned in the Faster R-CNN paper
        self.spatial_scale = spatial_scale
        self.use_cudnn = use_cudnn

        # Generate generic anchors
        # NOTE: Commented away since the anchors are now passed as an argument
        # self.anchors = generate_anchors()

    def check_type_forward(self, in_types):
        # TODO: Also check the dimensions of the given anchors
        type_check.expect(
            in_types.size() == 4,
            in_types[0].shape[2] == in_types[1].shape[2],  # e.g. 40
            in_types[0].shape[3] == in_types[1].shape[3],  # e.g. 27
            in_types[2].shape[1] == 4  # Anchor dimensions
        )

    def forward_cpu(self, inputs):
        raise NotImplementedError('forward_cpu')

    def forward_gpu(self, inputs):
        # Parse the input
        xp = cuda.get_array_module(*inputs)
        cls, bbox, anchors, gt_boxes = inputs
        width, height = cls.shape[-2:]

        print('---------- FORWARD ----------')
        print('Lambda: {}'.format(self.lbd))
        print('Spatial scale: {}'.format(self.spatial_scale))
        print('bbox.shape: {}'.format(bbox.shape))
        print('cls.shape: {}'.format(cls.shape))
        print('Downsampled width: {}'.format(width))
        print('Downsampled height: {}'.format(height))
        print('Ground truth boxes (target): {}'.format(gt_boxes))
        print('Inside Anchors: {}'.format(anchors.shape))

        feat_stride = 1 / self.spatial_scale
        print('feat_stride: {}'.format(feat_stride))

        # Bounding box labels, 1 is positive, 0 is negative, -1 is ignored
        labels = xp.empty((len(anchors), ), dtype=xp.float32)
        labels.fill(-1)

        overlaps = iouutils.ious(anchors, gt_boxes)

        print('IOUs of Inside Anchors and Ground Truth Boxes')
        print(overlaps)
        print('IOU Overlap Non-Zero Counts: {}'.format(xp.count_nonzero(overlaps)))
        print('Ground Truth Boxes Shape: {}'.format(overlaps.shape))

        # TODO: Continue here...

        # Select the ground truth box with highest IOU for each anchor
        argmax_overlaps = overlaps.argmax(axis=1)
        print('Highest IOU Ground Truth Index for each Anchor')
        print(argmax_overlaps)
        print('     #non-zeros: {}'.format(xp.count_nonzero(argmax_overlaps)))
        max_overlaps = overlaps.take((xp.arange(len(anchors)), argmax_overlaps)) # TODO Or use None to index all elements?
        print('Top Overlaps')
        print(max_overlaps)

        return (1,1) # Always a tuple, e.g. y, for all methods

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError('backward_cpu')

    def backward_gpu(self, inputs, grad_outputs):
        raise NotImplementedError('backward_gpu')


def multitask(cls, bbox, anchors, t, lbd=10, spatial_scale=0.0625, use_cudnn=True): # 0.0625 = 1/16, e.g. 4 max pooling layers with size and stride of 2.
    return MultiTask(lbd, spatial_scale, use_cudnn)(cls, bbox, anchors, t)
