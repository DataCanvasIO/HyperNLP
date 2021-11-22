from hypernlp.framework_config import Config
if Config.framework == 'tensorflow':
    import tensorflow as tf
    from tensorflow.keras import backend as K
elif Config.framework == 'pytorch':
    import torch.nn as nn
    import torch
else:
    raise TypeError("Unsupported framework: '{}'".format(Config.framework))


def tf_focal_loss(y_pred, y_true, gamma=2.0, alpha=0.8):
    y_true = tf.cast(y_true, tf.int64)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def tf_ce_loss(y_pred, y_true):
    if y_pred.shape[-1] == 1:
        y_true = tf.cast(y_true, dtype=float)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    else:
        y_true = tf.one_hot(y_true, y_pred.shape[-1])
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(loss)


def pt_focal_loss(y_pred, y_true, gamma=2.0, alpha=0.8):
    y_true = torch.reshape(y_true, (y_true.shape[0], 1)).to(torch.float)
    bce_loss = nn.BCELoss(reduce=False)(y_pred, y_true)
    pt = torch.exp(-bce_loss)
    loss = alpha * (1 - pt) ** gamma * bce_loss
    return torch.mean(loss)


def pt_ce_loss(y_pred, y_true):
    if y_pred.shape[1] == 1:
        y_true = torch.reshape(y_true, (y_true.shape[0], 1)).to(torch.float)
        return nn.BCELoss()(y_pred, y_true)
    else:
        return nn.CrossEntropyLoss()(y_pred, y_true)


def ce_loss(y_pred, y_true):
    if Config.framework == "tensorflow":
        return tf_ce_loss(y_pred, y_true)
    elif Config.framework == "pytorch":
        return pt_ce_loss(y_pred, y_true)
    else:
        raise ValueError("Unsupported framework: {}".format(Config.framework))


def focal_loss(y_pred, y_true):
    if Config.framework == "tensorflow":
        return tf_focal_loss(y_pred, y_true)
    elif Config.framework == "pytorch":
        return pt_focal_loss(y_pred, y_true)
    else:
        raise ValueError("Unsupported framework: {}".format(Config.framework))
