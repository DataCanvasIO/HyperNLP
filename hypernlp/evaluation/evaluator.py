import abc
import sys

import torch
from hypernlp.config import Config


class Evaluator(object):
    def __init__(self, model, validate_data, formulas):
        self.model = model
        self.validate = validate_data
        self.formulas = formulas

    @abc.abstractmethod
    def eval(self):
        pass


class PairWiseEvaluator(Evaluator):
    def __init__(self, model, validate_data, formulas):
        super(PairWiseEvaluator, self).__init__(model, validate_data, formulas)

    def eval(self):
        pred_res = []
        for i in range(self.validate.epoch_length):
            batch_data = self.validate.get_batch_data()
            if Config.framework == "tensorflow":
                pred = self.model(batch_data[:3], training=False).numpy()
                true = batch_data[-1].numpy()
            elif Config.framework == "pytorch":
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(batch_data[:3]).data.cpu().numpy()
                    true = batch_data[-1].data.cpu().numpy()
            else:
                raise ValueError("Unsupported framework: {}".format(Config.framework))
            for d in range(self.validate.batch_size):
                pred_res.append([pred[d], true[d]])
            sys.stdout.write("Processing evaluation: {}/{}".format(i, self.validate.epoch_length) + '\r')
            sys.stdout.flush()

        indicators = {}
        for formula in self.formulas:
            indicator, name = formula(pred_res)
            indicators[name] = indicator
        return indicators
