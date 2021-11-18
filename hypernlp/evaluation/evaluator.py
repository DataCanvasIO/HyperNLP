import abc
import sys
import json

import torch
import tensorflow as tf
from hypernlp.nlp.dataset import DatasetBase
from hypernlp.config import Config


class Evaluator(object):
    def __init__(self, validate_data, formulas):
        assert isinstance(validate_data, DatasetBase)
        self.validate = validate_data
        self.formulas = formulas

    @abc.abstractmethod
    def eval(self, model):
        raise NotImplementedError

    @abc.abstractmethod
    def run_process(self, batch_data, model):
        raise NotImplementedError


class PairWiseEvaluator(Evaluator):
    def __init__(self, validate_data, formulas):
        super(PairWiseEvaluator, self).__init__(validate_data, formulas)

    def run_process(self, batch_data, model):
        inputs, true = batch_data[:3], batch_data[-1]
        pred = model(inputs, training=False)
        true = batch_data[-1]
        return pred, true

    def eval(self, model):
        pred_res = []
        if Config.framework == "tensorflow":
            @tf.function
            def distributed_train_step(batch_data):
                output = Config.strategy.run(self.run_process, args=(batch_data, model,))
                return Config.strategy.experimental_local_results(output)

        for i in range(self.validate.epoch_length):
            batch_data = self.validate.get_batch_data()
            if Config.framework == "tensorflow":
                output = distributed_train_step(next(iter(batch_data)))
                pred, true = [], []
                for dev in output:
                    for d in dev[0].numpy():
                        pred.append(d)
                    for d in dev[1].numpy():
                        true.append(d[0])
            elif Config.framework == "pytorch":
                model.eval()
                with torch.no_grad():
                    pred = model(batch_data[:3]).data.cpu().numpy()
                    true = batch_data[-1].data.cpu().numpy()
            else:
                raise ValueError("Unsupported framework: {}".format(Config.framework))
            for d in range(self.validate.batch_size):
                pred_res.append([pred[d], true[d]])
            sys.stdout.write("Processing evaluation: {}/{}".format(i, self.validate.epoch_length) + '\r')
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        indicators = {}
        for formula in self.formulas:
            indicator, name = formula(pred_res)
            indicators[name] = indicator
        # return indicators
        return json.dumps(indicators, indent=4, ensure_ascii=False)


class MultiPairWiseEvaluator(Evaluator):
    def __init__(self, validate_data, formulas):
        super(MultiPairWiseEvaluator, self).__init__(validate_data, formulas)

    def eval(self, model):
        pass

    def run_process(self, batch_data, model):
        pass



