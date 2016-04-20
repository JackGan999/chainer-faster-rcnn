from chainer.cuda import cupy


def meshgrid(*xi):
    """Simplified implementation of numpy.meshgrid using cupy."""
    s0 = (1, 1)
    x, y = xi

    output = [cupy.asanyarray(x_tmp).reshape(s0[:i] + (-1,) + s0[i + 1::])
              for i, x_tmp in enumerate(xi)]
    # TODO: Alternatives to list comprehension?
    shape = [x.size for x in output]

    # Switch first and second axis
    fst_new_shape = (1, len(x))
    snd_new_shape = (len(y), 1)
    output[0] = output[0].reshape((fst_new_shape))
    output[1] = output[1].reshape((snd_new_shape))
    shape[0], shape[1] = shape[1], shape[0]

    mult_fact = cupy.ones(shape, dtype=int)

    # TODO: Alternatives to list comprehension?
    return [x * mult_fact for x in output]
