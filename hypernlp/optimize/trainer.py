import abc

import tensorflow as tf

from hypernlp.config import Config
from utils.logger import logger
from utils.string_utils import home_path


class Trainer(object):

    def __init__(self, model, optimzier, train_processers, epochs, epoch_length, evaluator, save_model_path):
        # with Config.strategy.scope():
        self.model = model
        self.__optimizer = optimzier
        self.optimizer = self.__optimizer()
        self.epochs = epochs
        self.epoch_length = epoch_length
        self.evaluator = evaluator
        self.save_path = save_model_path
        self.train_processers = train_processers
        assert isinstance(self.train_processers, list) or isinstance(self.train_processers, tuple)

    @abc.abstractmethod
    def run_optimization(self):
        pass

    def model_save_epoch(self, epoch):
        if not self.save_path[-1] == '/':
            raise SystemError("Checkpoints saving path should be end with '/'!")
        if Config.framework == "tensorflow":
            return self.save_path + "{}".format(epoch) + '.h5'
        elif Config.framework == "pytorch":
            return self.save_path + "{}".format(epoch) + '.pth'
        else:
            raise TypeError("Unsupported framework: '{}'".format(Config.framework))

    def checkpoint_save_epoch(self):
        if self.save_path is None:
            self.save_path = home_path() + 'hypernlp/optimize/checkpoints/'
        if not self.save_path[-1] == '/':
            raise SystemError("Checkpoints saving path should be end with '/'!")
        if Config.framework == "tensorflow":
            return self.save_path + 'checkpoint.h5', self.save_path + 'optimizer_cpt.h5'
        elif Config.framework == "pytorch":
            return self.save_path + 'checkpoint.pth', self.save_path + 'optimizer_cpt.pth'
        else:
            raise TypeError("Unsupported framework: '{}'".format(Config.framework))

    '''
    Custom definition?  User data input setting!
    '''
    def train(self):
        try:
            # if Config.framework == "tensorflow":
            #     tf.config.optimizer.set_jit(True)
            #     tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
            for epoch in range(self.epochs):
                for it in range(self.epoch_length):
                    loss = self.run_optimization()
                    logger.info("epoch[{}:{}/{}] loss: {}".format(epoch, it, self.epoch_length, loss))
                if self.save_path is not None:
                    self.model.model_save(self.model_save_epoch(epoch))
                if self.evaluator is not None:
                    logger.info("epoch[{}] evaluation: {}".format(epoch, self.evaluator.eval()))
        except KeyboardInterrupt:
            logger.info('manually interrupt, try saving model for now...')
            model_cpt, optim_cpt = self.checkpoint_save_epoch()
            self.model.model_save(model_cpt)
            self.__optimizer.save(optim_cpt)
            logger.info('model saved.')


class TFTrainer(Trainer):
    def __init__(self, model, losses, optimzier, epochs, epoch_length, evaluator, save_model_path):
        super(TFTrainer, self).__init__(model, losses, optimzier, epochs, epoch_length, evaluator, save_model_path)

        # register distribute_functions
        self.distibute_functions = []
        for i in range(len(self.train_processers)):
            @tf.function
            def distributed_train_step(tp_index, batch_data):
                per_replica_losses = Config.strategy.run(self.run_processer,
                                                         args=(self.train_processers[tp_index], batch_data, tp_index,))
                return Config.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                              axis=None)

            self.distibute_functions.append(distributed_train_step)

    def run_processer(self, tp, batch_data, index):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            loss = tp.step(self.model, batch_data, index)

            # Variables to update, i.e. trainable variables.
            trainable_variables = self.model.trainable_variables

            # Compute gradients.
            gradients = g.gradient(loss, trainable_variables)

            # Update W and b following gradients.
            self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss

    def run_optimization(self):
        total_loss = 0
        for index in range(0, len(self.train_processers)):
            batch_data = next(iter(self.train_processers[index].dataset.get_batch_data()))
            total_loss += self.distibute_functions[index](index, batch_data)
        return total_loss


class PTTrainer(Trainer):

    def __init__(self, model, optimzier, train_processers, epochs, epoch_length, evaluator, save_model_path):
        super(PTTrainer, self).__init__(model, optimzier, train_processers, epochs,
                                        epoch_length, evaluator, save_model_path)

    def run_optimization(self):
        self.model.train()
        self.optimizer.zero_grad()

        loss = self.train_processers[0].step(
            self.model, self.train_processers[0].dataset.get_batch_data(), 0)
        for index in range(1, len(self.train_processers)):
            loss += self.train_processers[index].step(
                self.model, self.train_processers[index].dataset.get_batch_data(), index)
        # Compute gradients.
        loss.backward()

        # Update W and b following gradients.
        self.optimizer.step()
        return loss.data.cpu().numpy()


def trainer(model, optimzier, train_processers, epochs, epoch_length, evaluator=None, save_model_path=None):
    if Config.framework == "tensorflow":
        return TFTrainer(model, optimzier, train_processers, epochs, epoch_length, evaluator, save_model_path)
    elif Config.framework == "pytorch":
        return PTTrainer(model, optimzier, train_processers, epochs, epoch_length, evaluator, save_model_path)
    else:
        raise ValueError("Unsupported framework: {}".format(Config.framework))


if __name__ == '__main__':
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

