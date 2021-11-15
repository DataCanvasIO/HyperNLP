import torch.nn as nn
import torch
from utils.logger import logger
from utils.string_utils import home_path
from transformers import BertConfig, BertModel, BertForPreTraining
from hypernlp.config import *


class PTModelBase(nn.Module):

    def __init__(self, bert_embedding):
        super(PTModelBase, self).__init__()
        if not isinstance(bert_embedding, nn.Module):
            raise TypeError("PTModel only accept nn.Module input!")
        self.bert_base = bert_embedding


class PTModel(nn.DataParallel):

    def __init__(self, pt_model):
        super(PTModel, self).__init__(pt_model)
    
    def model_save(self, path):
        torch.save(self.module.state_dict(), path)
        logger.info("Finish saving model to: '{}'.".format(path))

    def model_load(self, path):
        self.module.load_state_dict(torch.load(path), strict=False)
        logger.info("Finish loading model from: '{}'.".format(path))


def create_model(model_config):
    bert_config = BertConfig.from_pretrained(home_path() +
                                             model_config["BASE_MODEL_PATH"], output_hidden_states=True)
    bert_as_encoder = BertModel.from_pretrained(home_path() +
                                                model_config["BASE_MODEL_PATH"], config=bert_config)
    return bert_as_encoder


def create_pretraining_model(model_config):
    bert_config = BertConfig.from_pretrained(home_path() +
                                             model_config["BASE_MODEL_PATH"], output_hidden_states=True)
    bert_as_encoder = BertForPreTraining.from_pretrained(home_path() +
                                                model_config["BASE_MODEL_PATH"], config=bert_config)
    return bert_as_encoder