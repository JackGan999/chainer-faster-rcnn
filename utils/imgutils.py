import random
import numpy as np
import cv2 as cv


# Basic NumPy colors
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)


def draw_empty(width, height):
    """Create an empty black image with the given width and height."""
    num_channels = 3
    # Note the order of the width and height below
    return np.zeros((height, width, num_channels), np.uint8)


def draw_line(img, x1, y1, x2, y2, color=white, thickness=1):
    """Draw a line on the given image."""
    x1, y1, x2, y2 = map(_to_int, [x1, y1, x2, y2])
    cv.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_box(img, x1, y1, x2, y2, color=None, thickness=1):
    """Draw a rectangle on the given image."""
    if color is None:
        color = rnd_color()
    x1, y1, x2, y2 = map(_to_int, [x1, y1, x2, y2])
    cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def write_img(filename, img):
    """Save the given image to a file."""
    print('imgutil: Saving {} with shape {}'.format(filename, img.shape))
    cv.imwrite(filename, img)


def rnd_color():
    """Return a random RGB color."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


def _to_int(x):
    """Try to parse the given value as an int if it already isn't."""
    return x if isinstance(x, int) else int(x)
