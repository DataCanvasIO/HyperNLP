import tensorflow.keras as keras
import tensorflow as tf

from utils.logger import logger
from utils.string_utils import home_path
from transformers import BertConfig, TFBertModel, TFBertForPreTraining
from hypernlp.config import Config


class TFModel(keras.Model):

    def __init__(self, bert_embedding):
        super(TFModel, self).__init__()
        if not isinstance(bert_embedding, keras.Model):
            raise TypeError("TFModel only accept tf.keras.Model input!")
        self.bert_base = bert_embedding

    def model_save(self, path):
        tf.saved_model.save(
            self, path, signatures=None, options=None
        )
        logger.info("Finish saving model to: '{}'.".format(path))

    def model_load(self, path):
        self.load_weights(path)
        logger.info("Finish loading model from: '{}'.".format(path))


def create_model(model_config):
    bert_config = BertConfig.from_pretrained(home_path() +
                                             model_config["BASE_MODEL_PATH"], output_hidden_states=True)
    with Config.strategy.scope():
        bert_as_encoder = TFBertModel.from_pretrained(home_path() +
                                                      model_config["BASE_MODEL_PATH"], config=bert_config)
    return bert_as_encoder


def create_pretraining_model(model_config):
    bert_config = BertConfig.from_pretrained(home_path() +
                                             model_config["BASE_MODEL_PATH"], output_hidden_states=True)
    with Config.strategy.scope():
        bert_as_encoder = TFBertForPreTraining.from_pretrained(home_path() +
                                                      model_config["BASE_MODEL_PATH"], config=bert_config)
    return bert_as_encoder