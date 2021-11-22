import abc
import sys

from hypernlp.framework_config import Config
if Config.framework == 'tensorflow':
    import tensorflow as tf
elif Config.framework == 'pytorch':
    import torch
else:
    raise TypeError("Unsupported framework: '{}'".format(Config.framework))
from hypernlp.nlp.dataset import DatasetBase


class Predictor(object):
    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def predict(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def run_process(self, batch_data, model):
        raise NotImplementedError


class NLPPredictor(Predictor):
    def __init__(self, model):
        super(NLPPredictor, self).__init__(model)

    def run_process(self, batch_data, model):
        inputs = batch_data[:3]
        pred = model(inputs, training=False)
        return pred

    def predict(self, data):
        assert isinstance(data, DatasetBase)
        pred_res = []
        if Config.framework == "tensorflow":
            @tf.function
            def distributed_train_step(batch_data):
                output = Config.strategy.run(self.run_process, args=(batch_data, self.model,))
                return Config.strategy.experimental_local_results(output)

        for i in range(data.epoch_length):
            batch_data = data.get_batch_data()
            if Config.framework == "tensorflow":
                output = distributed_train_step(next(iter(batch_data)))
                pred = []
                for dev in output:
                    for d in dev.numpy():
                        pred.append(d)
            elif Config.framework == "pytorch":
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(batch_data[:3]).data.cpu().numpy()
            else:
                raise ValueError("Unsupported framework: {}".format(Config.framework))
            for d in range(data.batch_size):
                pred_res.append([pred[d]])
            sys.stdout.write("Processing evaluation: {}/{}".format(i, data.epoch_length) + '\r')
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return pred_res
