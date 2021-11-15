import torch
import torch.optim as pt_optim
from utils.logger import logger

from hypernlp.dl_framework_adaptor.optimizer.optimizer_base import OptimBase


class PTOptim(OptimBase):

    def __init__(self, model, optimizer_type, param):
        super().__init__(model, optimizer_type, param)

        if optimizer_type == "sgd":
            self.optimizer = pt_optim.SGD(params=self.model.parameters(), lr=param["lr"],
                                          momentum=param["momentum"], weight_decay=param['weight_decay'])
        elif optimizer_type == "adam":
            self.optimizer = pt_optim.Adam(params=self.model.parameters(), lr=param["lr"],
                                           betas=param["betas"],
                                           eps=param["epsilon"],
                                           amsgrad=param["amsgrad"],
                                           weight_decay=param['weight_decay'])
        elif optimizer_type == "rmsprop":
            self.optimizer = pt_optim.RMSprop(params=self.model.parameters(), lr=param["lr"],
                                              alpha=param["alpha"],
                                              momentum=param["momentum"],
                                              eps=param["epsilon"],
                                              centered=param["centered"],
                                              weight_decay=param['weight_decay'])
        elif optimizer_type == "adagrad":
            self.optimizer = pt_optim.Adagrad(params=self.model.parameters(), lr=param["lr"],
                                              lr_decay=param["lr_decay"],
                                              initial_accumulator_value=param["initial_accumulator_value"],
                                              eps=param["epsilon"],
                                              weight_decay=param["weight_decay"])
        else:
            raise ValueError("Unsupported optimizer: '{}'".format(optimizer_type))

    def call_back_evo(self, evaluate, validata_data):
        self.step()
        evaluate(self.model, validata_data, self.param)

    def step(self):
        self.optimizer.step()

    def save(self, path):
        torch.save(self.optimizer.state_dict(), path)
        logger.info("Finish saving optimizer to: '{}'.".format(path))

    def load(self, path):
        self.optimizer.load_state_dict(torch.load(path))
        logger.info("Finish loading optimizer from: '{}'.".format(path))

    def __call__(self, *args, **kwargs):
        return self.optimizer
