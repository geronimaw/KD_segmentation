import tensorflow as tf

def mmd_loss(X, Y, sigmas=(1,), wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    X = tf.linalg.norm(X, axis=[1,2])
    Y = tf.linalg.norm(Y, axis=[1,2])

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma**2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return tf.reduce_sum(K_XX + K_YY - 2*K_XY)


# def mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
#     m = tf.cast(K_XX.get_shape()[0], tf.float32)
#     n = tf.cast(K_YY.get_shape()[0], tf.float32)
#
#     if biased:
#         mmd2 = (tf.reduce_sum(K_XX) / (m * m)
#               + tf.reduce_sum(K_YY) / (n * n)
#               - 2 * tf.reduce_sum(K_XY) / (m * n))
#     else:
#         if const_diagonal:
#             trace_X = m * const_diagonal
#             trace_Y = n * const_diagonal
#         else:
#             trace_X = tf.linalg.trace(K_XX)
#             trace_Y = tf.linalg.trace(K_YY)
#
#         # mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
#         #       + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
#         #       - 2 * tf.reduce_sum(K_XY) / (m * n))
#         mmd2 = K_XX + K_YY - 2*K_XY
#
#     return tf.reduce_sum(mmd2)