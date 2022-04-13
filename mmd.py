import tensorflow as tf
from tensorflow import *
from tensorflow import transpose as t
from tensorflow.linalg import matmul as mm

def linear_kernel(a,b):
    return tf.linalg.matmul(t(a), b)


def MMD(x, y, batch, kernel="multiscale"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    xx = mm(x[:, :, :, :batch], t(x[:, :, :, :batch]))
    yy = mm(y[:, :, :, :batch], t(y[:, :, :, :batch]))
    zz = mm(x[:, :, :, :batch], t(y[:, :, :, :batch]))

    # rx = broadcast_to(expand_dims(tf.linalg.diag_part(xx), axis=-1), xx.shape)
    # ry = broadcast_to(expand_dims(tf.linalg.diag_part(yy), axis=-1), yy.shape)
    rx = xx
    ry = yy

    dxx = t(rx) + rx - 2 * xx  # Used for A in (1)
    dyy = t(ry) + ry - 2 * yy  # Used for B in (1)
    dxy = t(rx) + ry - 2 * zz  # Used for C in (1)

    XX = zeros((batch, batch, batch, batch))
    YY = XX
    XY = XX

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    # elif kernel == "rbf":
    #
    #     bandwidth_range = [10, 15, 20, 50]
    #     for a in bandwidth_range:
    #         XX += torch.exp(-0.5 * dxx / a)
    #         YY += torch.exp(-0.5 * dyy / a)
    #         XY += torch.exp(-0.5 * dxy / a)

    return math.reduce_mean(XX + YY - 2. * XY)
    # return dxx
    # # return xx
    # # return x + y

    # return xx