import os
from hypernets.conf import configure, Configurable, Int, String, Bool


@configure()
class Config(Configurable):
    # GPU
    CUDA_VISIBLE_DEVICES = String("0, 1", allow_none=False,
                                  help='"GPU VISIBLE" setting for GPU use.').tag(config=True)
    GPUS = Int(2, min=1,
               help='').tag(config=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES.default_value
    framework = String("pytorch", allow_none=False,
                       help='Framework selection for "tensorflow" or "pytorch".').tag(config=True)

    if framework.default_value == "tensorflow":
        import tensorflow as tf
        devices = ["/gpu:{}".format(d) for d in CUDA_VISIBLE_DEVICES.default_value.split(',')]
        strategy = tf.distribute.MirroredStrategy(devices)
        print('Number of devices: {}{}'.format(strategy.num_replicas_in_sync, devices))
        USE_XLA = Bool(False, help='Use XLA to accelerate calculate in tensorflow.').tag(config=True)