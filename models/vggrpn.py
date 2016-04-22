import chainer
import chainer.links as L
import chainer.functions as F
from functions.loss.multitask import multitask

F.multitask = multitask


class VGGRPN(chainer.Chain):
    """VGGNet
    - It takes (224, 224, 3) sized image as input
    """

    def __init__(self):
        super(VGGRPN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            # RPN. See models/coco/VGG16/faster_rcnn_end2end/train.prototext
            # for reference
            rpn_conv=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            # RPN Classification (foreground/background) Sibling
            # (Kernel size = 1, a linear mapping)
            # 18 = 2 (bg/fg) * 9 (anchors)
            rpn_cls=L.Convolution2D(512, 18, 1, stride=1, pad=0),

            # RPN Bounding Box Prediction Sibling
            # (Kernel size = 1, a linear mapping)
            # 36 = 4 (x, y, w, h) * 9 (anchors)
            rpn_bbox=L.Convolution2D(512, 36, 1, stride=1, pad=0),

            # TODO: Remove the following layers if they aren't used
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = True
        self.k = 9  # Number of achors

    def __call__(self, x, t, anchors):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        # h = F.max_pooling_2d(h, 2, stride=2)

        # RPN
        # h = F.relu(self.rpn_conv(h))
        h = F.relu(self.rpn_conv(h))

        # h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        # h = self.fc8(h)

        if self.train:
            print('self.train == True')
            # TODO Need to compute the loss and the acc here
            # TODO Create a new loss function class (multitask.py,
            # function.Function) and set self.loss = F.multitask(h, t)

            cls = F.softmax(self.rpn_cls(h))

            print('Computed cls softmax')

            # TODO For now, create a target t by copying the values from the
            # cls to test the smooth L1 loss
            # t = Variable(cls.data.copy())
            # self.smooth_l1_loss = huber_loss(cls, t, delta=1)
            # print('Smooth L1 loss shape: {}'.format(self.smooth_l1_loss
            # .data.shape))

            bbox = self.rpn_bbox(h)

            self.loss = F.multitask(cls, bbox, anchors, t)

            return self.loss

            # self.loss = F.softmax_cross_entropy(h, t)
            # self.acc = F.accuracy(h, t)
            # return self.loss
        else:
            print('self.train == False')
            self.pred = self.rpn_cls(h), self.rpn_bbox(h)
            # self.pred = F.softmax(h)
            return self.pred
