from hypernlp.framework_config import Config
if Config.framework == "tensorflow":
    import tensorflow.keras.optimizers as tf_optim
    from utils.logger import logger

    from hypernlp.dl_framework_adaptor.optimizer.optimizer_base import OptimBase
    from hypernlp.framework_config import Config


    class TFOptim(OptimBase):

        def __init__(self, model, optimizer_type, param):
            super().__init__(model, optimizer_type, param)

            if optimizer_type == "sgd":
                self.optimizer = tf_optim.SGD(learning_rate=param["lr"],
                                              momentum=param["momentum"])
            elif optimizer_type == "adam":
                self.optimizer = tf_optim.Adam(learning_rate=param["lr"],
                                               beta_1=param["betas"][0],
                                               beta_2=param["betas"][1],
                                               epsilon=param["epsilon"],
                                               amsgrad=param["amsgrad"])
            elif optimizer_type == "rmsprop":
                self.optimizer = tf_optim.RMSprop(learning_rate=param["lr"],
                                                  rho=param["alpha"],
                                                  momentum=param["momentum"],
                                                  epsilon=param["epsilon"],
                                                  centered=param["centered"])
            elif optimizer_type == "adagrad":
                self.optimizer = tf_optim.Adagrad(learning_rate=param["lr"],
                                                  decay=param["lr_decay"],
                                                  initial_accumulator_value=param["initial_accumulator_value"],
                                                  epsilon=param["epsilon"],
                                                  weight_decay=param["weight_decay"])
            else:
                raise ValueError("Unsupported optimizer: '{}'".format(optimizer_type))

        '''
        tf optimizer save&load need fixed
        '''
        def save(self, path):
            self.optimizer.get_weights()
            # self.optimizer.(path)
            logger.info("Finish saving optimizer to: '{}'.".format(path))

        def load(self, path):
            self.optimizer.load_weights(path)
            logger.info("Finish loading optimizer from: '{}'.".format(path))

        def __call__(self, *args, **kwargs):
            return self.optimizer