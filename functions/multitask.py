from chainer import function


class MultiTask(function.Function):

  def __init__(self, use_cudnn=True):
    self.use_cudnn = use_cudnn

  def check_type_forward(self, in_types):
    # TODO
    print('Not yet implemented')

  def forward_gpu(self, inputs):
    x, t = inputs
    # TODO
    print('Not yet implemented')

  def backward_gpu(self, inputs, grad_outputs):
    x, t = inputs
    # TODO
    print('Not yet implemented')


def multitask(x, t, use_cudnn=True):
  return MultiTask(use_cudnn)(x, t)
