import os
import sys
import numpy as np
import six

# NOTE: Not tested with most recent code.

# Dataset base directory, set to development directory for quick testing
basedir = "./data/coco/samples/"


def create_from_mscoco(filename, shape, size):
    """Save an MSCOCO dataset representational file to disk with the name
    specified by the parameter.
    """
    if not os.path.exists(filename):
        print('Processing MSCOCO to create {}, this might take a while...'
              .format(filename))
        data_train, target_train = load_reshape(1, shape, size)

        coco = {}
        coco['data'] = np.append(data_train, target_train, axis=0)

        with open(filename, 'wb') as output:
            six.moves.cPickle.dump(coco, output, -1)
    else:
        print('Found {}'.format(filename))


def load_reshape(num, shape, size):
    """Load the MNIST gzip files from disk and reshape the data."""
    shape = tuple([num, *shape])
    data = np.zeros(num * size, dtype=np.uint8).reshape(shape)
    target = None

    # TODO
    # target = np.zeros(num, dtype=np.uint8).reshape((num, ))

    # For each file (.jpg) in the directory
    for file in os.listdir(basedir):
        with open(os.path.join(basedir, file), 'r') as image:
            # image.read(16)
            print(image)
            image.read()
            for i in six.moves.range(num):
                print(i)
                # target[i] = ord(f_labels.read(1))
                for j in six.moves.range(shape[1]):  # For each pixel in width
                    print(j)
                    for k in six.moves.range(shape[2]):
                        # For each pixel in height
                        print(k)
                        data[i, 0, j, k] = ord(image.read(1))
                        print(data)

    # with gzip.open(images, 'rb') as f_images,\
    #         gzip.open(labels, 'rb') as f_labels:
    #           f_images.read(16)
    #           f_labels.read(8)

    return data, target


if __name__ == '__main__':
    """Go through all predownloaded images of the MSCOCO dataset, pack them
    into pickled files so that they can be easily read durint training. Notice
    that the large size of the file (estimated to be around 10-15GB) assumes
    that the hardware running this script has sufficient amounts of memory.

    The data is reshaped in to 2-dimensional data with annotations with 3
    channels representing the RGC color channels.
    """

    pklfilename = 'mscoco.pkl'
    if os.path.exists(pklfilename):
        print('The MSCOCO dataset is already reshaped and processed')
        sys.exit()

    print('Preparing the MSCOCO dataset...')

    width = 640
    height = 480
    num_channels = 3
    size = width * height * num_channels
    dim = (num_channels, width, height)
    create_from_mscoco(pklfilename, dim, size)
    print('Done preparing the MSCOCO dataset')
