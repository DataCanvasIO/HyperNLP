import tensorflow as tf
import torch
import random
import time

from hypernlp.framework_config import Config


def environment_check():
    random.seed(int(time.time()))
    print('Environment check and config:')

    if Config.framework == "tensorflow":
        status = tf.test.is_gpu_available()
        print('TensorFlow Version:', tf.__version__)
        print('GPU status:', status)
        print('Environment check and config finished.')

    elif Config.framework == "pytorch":
        print('pytorch Version:', torch.__version__)
        status = torch.cuda.is_available()
        print('GPU status:', status)
        print('Environment check and config finished.')
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))


def is_gpu_available():

    # specify GPU usage
    if Config.framework == "tensorflow":
        status = tf.test.is_gpu_available()
    elif Config.framework == "pytorch":
        status = torch.cuda.is_available()
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))

    return status

