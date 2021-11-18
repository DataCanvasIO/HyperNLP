import os
import tensorflow as tf


class Config:
    # GPU
    CUDA_VISIBLE_DEVICES = "0, 1"
    GPUS = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    framework = "pytorch"#"tensorflow"#

    if framework == "tensorflow":
        devices = ["/gpu:{}".format(d) for d in CUDA_VISIBLE_DEVICES.split(',')]
        strategy = tf.distribute.MirroredStrategy(devices)
        print('Number of devices: {}{}'.format(strategy.num_replicas_in_sync, devices))
