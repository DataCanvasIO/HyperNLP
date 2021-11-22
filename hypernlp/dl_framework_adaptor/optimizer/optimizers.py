from hypernlp.framework_config import Config
if Config.framework == 'tensorflow':
    from hypernlp.dl_framework_adaptor.optimizer.tf_optimizers import TFOptim
elif Config.framework == 'pytorch':
    from hypernlp.dl_framework_adaptor.optimizer.pt_optimizers import PTOptim
else:
    TypeError("Unsupported framework: {}".format(Config.framework))
from utils.logger import logger


def optimizer(model, optimizer_type, param):
    logger.info('Start optimizing: ' + str(param))
    if Config.framework == "tensorflow":
        with Config.strategy.scope():
            optim = TFOptim(model, optimizer_type, param)
        return optim
    elif Config.framework == "pytorch":
        return PTOptim(model, optimizer_type, param)
    else:
        raise TypeError("Unsupported framework: {}".format(Config.framework))
