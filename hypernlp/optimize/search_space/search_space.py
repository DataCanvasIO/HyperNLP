from hypernlp.dl_framework_adaptor.optimizer.optimizer_base import *
from hypernlp.dl_framework_adaptor.optimizer.optimizers import *


class SearchSpace(object):

    # params is a dict that contains the Hyperparams
    # params = {"lr": Param([0.1, 0.01, 0.05]), "weight_decay": Param([0.01, 0.001, 0.0001])}
    def __init__(self, params):
        self.params = params
        self.param_mapping = {}

    def call_back_param_sample(self, param_status):
        pass
    

class OptimizerSearchSpace(SearchSpace):
    
    def __init__(self, optimizer_type, params):
        super(OptimizerSearchSpace, self).__init__(params)
        self.optimizer_type = optimizer_type

    def check_params(self):
        check_param(self.optimizer_type, self.params)

    def call_back_param_sample(self, param_status):
        param_token = optimizer_params(optimizer_type=self.optimizer_type)
        param = {}

        for token in param_token:
            param[token] = param_status.get(token)

        return param


if __name__ == '__main__':
    op = OptimizerSearchSpace('adam', None)
