from hypernlp.config import Config
import torch.nn as nn
import tensorflow as tf


class Compiler(object):

    def __init__(self, downstream_model, optimizer, loss, ):
        print("")


def tf_compiler(downstream_model):
    with Config.strategy.scope():
        downstream_model = downstream_model
    return downstream_model


def pt_compiler(downstream_model):
    class complied_model(nn.Module):

        def __init__(self, downstream_model, loss):
            super(complied_model, self).__init__(downstream_model, loss)
            self.downstream_model = downstream_model
            self.loss = loss



