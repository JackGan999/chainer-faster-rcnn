import unittest

from utils import imgutils
from utils import anchorutils


class BasicAnchorCreationTest(unittest.TestCase):

    def setUp(self):
        self.gpu = True
        self.anchorsize_err_tolerance = 0.2
        self.filename = 'anchors_basic.jpg'

    def test_anchor_sizes(self):
        """ Generate the basic 9 anchors and compute their sizes to make sure
        that the sizes of the anchors aren't too small or too big.
        """
        gpu = self.gpu
        anchorsize_err_tolerance = self.anchorsize_err_tolerance

        base_size = 16
        ratios = [0.5, 1, 2]
        scales = [8, 16, 32]

        # Possible sizes (areas), e.g. 128^2, 256^2, 512^2
        original_sizes = [(base_size * scale) ** 2 for scale in scales]

        anchors = anchorutils.generate_anchors(base_size, ratios, scales,
                                               gpu=gpu)

        for anchor in anchors:
            x1, y1, x2, y2 = anchor
            size = (x2 - x1) * (y2 - y1)

            # Compute the difference between the sizes of the generated anchor
            # and the original anchor sizes
            diffs = [abs(original_size - size)
                     for original_size in original_sizes]
            min_idx, min_val = min(enumerate(diffs), key=lambda p: p[1])

            self.assertTrue(min_val < original_sizes[min_idx] *
                            anchorsize_err_tolerance)

    def test_draw_anchors(self):
        """ Generate the basic 9 anchors and draw and then save them to a
        file.
        """
        gpu = self.gpu
        filename = self.filename

        img_width = 1024
        img_height = 1024
        img = imgutils.draw_empty(img_width, img_height)

        anchors = anchorutils.generate_anchors(gpu=gpu)

        for anchor in anchors:
            x1, y1, x2, y2 = anchor

            # Shift the anchor center to the image center
            x_shift = 0.5 * img_width
            y_shift = 0.5 * img_height
            x1 += x_shift
            y1 += y_shift
            x2 += x_shift
            y2 += y_shift

            anchor_color = imgutils.rnd_color()

            imgutils.draw_box(img, x1, y1, x2, y2, color=anchor_color)

        imgutils.write_img(filename, img)


if __name__ == '__main__':
    unittest.main()
