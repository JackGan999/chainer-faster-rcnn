import chainer
from chainer import Variable, optimizers, serializers
from models.vggrpn import VGGRPN
from mscoco import MSCOCO
from utils import anchorutils

# TODO: Hardcoded to use the GPU
xp = chainer.cuda.cupy


def load_model(filename, model):
    """Load the model with the file data."""
    print('Loading pretrained model...')
    try:
        serializers.load_hdf5(filename, model)
        print('Successfully loaded pretrained model')
    except OSError as err:
        print('OS error: {}'.format(err))
        print('Could not find a pretrained model. \
                Proceeding with a randomly initialized model.')


if __name__ == "__main__":
    print("Training the RPN...")
    model = VGGRPN()

    # TODO: Skip the model loading during test.
    # load_model('vggrpn.model', model)

    model.to_gpu()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    # Load a sample image, the image with id 233833
    coco = MSCOCO()
    coco.load_images('./data/coco/images/test')
    coco.load_annotations('./data/coco/annotations/233833_annotations.json')

    # Get the image and annotation data from the MSCOCO wrapper
    image, annotations = coco.image('233833')

    # Preprocess the annotation (ground truth boxes)
    gtboxes = []
    for annotation in annotations:
        x1, y1, w, h = annotation['bbox']
        gtbox = [x1, y1, x1 + w, y1 + h]
        gtboxes.append(gtbox)
    gtboxes = xp.array(gtboxes, dtype=xp.float32)

    print('Ground Truth Boxes')
    print(gtboxes)

    print(image.shape)
    img_width = image.shape[2]
    img_height = image.shape[3]

    # Optimization
    # Generate anchors once, assuming that the dimensions are the same,
    # reuse those anchors throughout the training
    anchors = anchorutils.generate_inside_anchors(
            img_width, img_height, feat_stride=16, allowed_offset=None,
            gpu=True)
    print('Anchors inside image with dimensions ({w}, {h}): {num_anchors}'
          .format(w=img_width, h=img_height, num_anchors=len(anchors)))
    print('Anchor array module: {}'.format(xp.get_array_module(anchors)))

    # Start the training
    for epoch in range(1):
        print('Epoch: {epoch}'.format(epoch=epoch))

        # TODO: All data used in during the  epoch should be transferred
        # to the GPU here to for performance resasons.

        for i in range(1):
            # Mini batch
            # x = Variable(xp.asarray(x_train))
            # t = Variable(xp.asarray(y_train[indexes[i : i + batchsize]]))
            x = Variable(xp.asarray(image))
            print("image.shape: {}".format(x.data.shape))
            t = Variable(gtboxes)
            model.zerograds()
            # TODO: Make sure we need to reinitialize the anchors here as
            # Chainer variables, otherwise do it once in the beginning
            anchors_var = Variable(anchors)
            loss = model(x, t, anchors_var)
            print('Loss: {}'.format(loss))
            # optimizer.update(model, x, t)
