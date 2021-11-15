import abc


def check_sgd_param(param):
    if 'lr' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('lr'))
    if 'momentum' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('momentum'))
    if 'weight_decay' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('weight_decay'))


def check_adam_param(param):
    if 'lr' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('lr'))
    if 'betas' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('betas'))
    if 'epsilon' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('epsilon'))
    if 'weight_decay' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('weight_decay'))
    if 'amsgrad' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('amsgrad'))


def check_rmsprop_param(param):
    if 'lr' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('lr'))
    if 'alpha' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('alpha'))
    if 'epsilon' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('epsilon'))
    if 'momentum' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('momentum'))
    if 'weight_decay' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('weight_decay'))
    if 'centered' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('centered'))


def check_adagrad_param(param):
    if 'lr' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('lr'))
    if 'lr_decay' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('lr_decay'))
    if 'epsilon' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('epsilon'))
    if 'initial_accumulator_value' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('initial_accumulator_value'))
    if 'weight_decay' not in param.keys():
        raise ValueError("Cannot find parameter: {}".format('weight_decay'))


def optimizer_params(optimizer_type):
    if optimizer_type == "sgd":
        return ["lr", "momentum", "weight_decay"]
    elif optimizer_type == "adam":
        return ["lr", "weight_decay", "betas", "epsilon", "amsgrad"]
    elif optimizer_type == "rmsprop":
        return ["lr", "weight_decay", "alpha", "epsilon", "momentum", "centered"]
    elif optimizer_type == "adagrad":
        return ["lr", "lr_decay", "initial_accumulator_value", "epsilon", "weight_decay"]
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_type))


def check_param(optimizer_type, param):
    if optimizer_type == "sgd":
        check_sgd_param(param)
    elif optimizer_type == "adam":
        check_adam_param(param)
    elif optimizer_type == "rmsprop":
        check_rmsprop_param(param)
    elif optimizer_type == "adagrad":
        check_adagrad_param(param)
    else:
        raise ValueError("Unsupported optimizer: {}".format(optimizer_type))


class OptimBase(object):

    def __init__(self, model, optimizer_type, param):
        self.model = model
        self.optimizer_type = optimizer_type
        self.param = param

        check_param(optimizer_type, param)
        # if Config.framework == "pytorch":
        #     self.optimizer = PTOptim(model, optimizer_type, param)
        # elif Config.framework == "tensorflow":
        #     self.optimizer = TFOptim(model, optimizer_type, param)
        # else:
        #     raise ValueError("Unsupported optimizer: '{}'".format(optimizer_type))

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass