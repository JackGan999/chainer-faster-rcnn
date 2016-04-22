import unittest

import numpy as np
from chainer.cuda import cupy as cp

from utils import imgutils
from utils import anchorutils


class TestAnchorUtils(unittest.TestCase):

    def setUp(self):
        self.gpu = True
        self.write_result_img = True

    def test_anchor_generation(self):
        gpu = self.gpu
        write_result_img = self.write_result_img

        img_width = 640
        img_height = 427
        feat_stride = 16  # 2^4 as in VGG16

        # Generate the anchors, for the given image properties
        # This should only be done once for our dataset for performance reasons
        anchors_inside = anchorutils.generate_inside_anchors(
            img_width, img_height, feat_stride=feat_stride,
            allowed_offset=None, gpu=gpu)

        # Assert that the returned list of anchors is either on the CPU
        # or the GPU depending on the given parameters
        arr_module = cp.get_array_module(anchors_inside)
        if gpu:
            self.assertTrue(arr_module == cp)
        else:
            self.assertTrue(arr_module == np)

        print('Anchors inside: {}'.format(len(anchors_inside)))

        img_area = img_width * img_height
        for anchor in anchors_inside:
            x1 = anchor[0]
            y1 = anchor[1]
            x2 = anchor[2]
            y2 = anchor[3]

            self.assertTrue(x1 >= 0)
            self.assertTrue(y1 >= 0)
            self.assertTrue(x2 < img_width)
            self.assertTrue(y2 < img_height)

            area = (x2 - x1) * (y2 - y1)

            self.assertTrue(area > 0)
            self.assertTrue(area <= img_area)

        # Save the image to disk if specified in the test
        if write_result_img:
            print('Saving image to disk...')
            img = imgutils.draw_empty(img_width, img_height)
            for anchor in anchors_inside:
                imgutils.draw_box(img, anchor[0], anchor[1], anchor[2],
                                  anchor[3])
            imgutils.write_img('test_anchor_generation.jpg', img)


if __name__ == '__main__':
    unittest.main()
