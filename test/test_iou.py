import unittest

import time
import math
import numpy as np
from chainer.cuda import cupy as cp
from utils import iouutils


class TestIouUtils(unittest.TestCase):

    def setUp(self):
        anchors = cp.arange(1000, dtype=cp.float32)
        gt_boxes = anchors * 1
        anchors = anchors.reshape(250, 4)
        gt_boxes = gt_boxes.reshape(250, 4)[:10]

        self.cpu_anchors = cp.asnumpy(anchors)
        self.cpu_gt_boxes = cp.asnumpy(gt_boxes)
        self.gpu_anchors = anchors
        self.gpu_gt_boxes = gt_boxes

    def test_acc_gpu(self):
        boxes = cp.array([[0, 0, 10, 10], [2, 3, 4, 5]], dtype=cp.float32)
        query_boxes = cp.array([[0, 0, 10, 10], [2, 2, 15, 15], [5, 5, 15, 15], [20, 20, 22, 30]], dtype=cp.float32)
        ious = iouutils.ious(boxes, query_boxes)
        print(ious)
        print(ious.shape)

    def test_speed_gpu(self):
        a = cp.array([[1,2,1,2], [3,4,3,4], [5,5,5,5]], dtype=cp.float32)
        b = cp.array([[10,2,10,2], [1,1,11,1]], dtype=cp.float32)
        ans = iouutils.ious(a, b)
        print(ans)
        print(ans.shape)

    # def test_gpu_speed(self):
    #     # TODO
    #     print('GPU')
    #     anchors = self.gpu_anchors
    #     gt_boxes = self.gpu_gt_boxes
    #     assert cp.get_array_module(anchors, gt_boxes).__name__ == 'cupy'
    #     # ious = bboxutils.ious_gpu(anchors, gt_boxes)
    #     start = time.clock()
    #     for gt_box in gt_boxes:
    #         for anchor in anchors:
    #             iou = bboxutils.iou_gpu(anchor, gt_box)
    #             # print(iou)
    #     end = time.clock()
    #     print('GPU Time: {} s'.format(end - start))

    # def test_gpu_acc(self):
    #     anchor = cp.array([10, 10, 20, 40], dtype=cp.float32)
    #     gt_box = cp.array([15, 15, 35, 45], dtype=cp.float32)
    #     # iou = bboxutils.iou_gpu(anchor, gt_box)
    #     # assert math.isclose(iou, 0.1613, rel_tol=1e-2)
    #     # print('IOU {}'.format(iou))

    #     anchors = cp.vstack((anchor, (anchor + 10), anchor))
    #     gt_bboxs = cp.vstack((gt_box, (anchor + 1.0)))

    #     for g in gt_bboxs:
    #         for a in anchors:
    #             print('----------------')
    #             print(g)
    #             print(a)
    #             print(bboxutils.iou_cpu(a, g))

    #     ious = bboxutils.ious_gpu(anchors, gt_bboxs)
    #     print('IOUS {}'.format(ious.shape))
    #     for iou in ious:
    #         print(iou)


if __name__ == '__main__':
    unittest.main()
