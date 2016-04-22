import os
import cv2
import numpy as np
import json
from utils import profiler


def _filename_from_id(id, group='train'):
    if isinstance(id, int):
        id = str(id)
    padding = '0' * (12 - len(id))
    return 'COCO_{}2014_{}.jpg'.format(group, padding + id)


class MSCOCO:

    def __init__(self):
        self.files = {}
        self.annotations = {}

    def load_annotations(self, path):
        with open(path) as jsonfile:
            decoded = json.load(jsonfile)
            for annotation in decoded['annotations']:
                filename = _filename_from_id(annotation['image_id'])
                if filename in self.annotations:
                    self.annotations[filename].append(annotation)
                else:
                    self.annotations[filename] = [annotation]

    def load_images(self, path):
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                self.files[f] = cv2.imread(os.path.join(path, f))

    def _load_images(self, path):
        """ Use later. Not used at the moment since we want to use dictionaries a
        nd not lists.
        """
        raise NotImplementedError('_load_images')

    def image(self, id, group='train', normalized=True):
        filename = _filename_from_id(id, group)
        image = self.files[filename]
        height, width, channels = image.shape

        # Reshape to Chainer preferred format,
        # i.e. (batch_size, n_chanels, width, height)
        image = np.array([image]).astype(np.float32) \
                  .reshape((1, channels, width, height))

        if normalized:
            image /= 255

        annotation = self.annotations[filename]
        return image, annotation


if __name__ == '__main__':
    # NOTE: Run a local test when it is run as main
    print('Memory usage (before): {} MB'
          .format(profiler.memory_usage(format='mb')))
    coco = MSCOCO()
    print('Loading images...')
    coco.load_images('./data/coco/images/test')
    print('Done loading images')
    print('Loading annotations...')
    coco.load_annotations('./data/coco/annotations/233833_annotations.json')
    print('Done loading annotations')
    print('Memory usage (after): {} MB'
          .format(profiler.memory_usage(format='mb')))
    image, annotations = coco.image('233833')
    print(image.shape)
    print(annotations)
    bboxs = [a['bbox'] for a in annotations]
    print(bboxs)
