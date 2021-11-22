from hypernlp.framework_config import Config
if Config.framework == "tensorflow":
    import abc
    import numpy as np
    import tensorflow.keras as keras
    import tensorflow as tf

    from utils.logger import logger
    from utils.string_utils import home_path
    from transformers import BertConfig, TFBertModel, TFBertForPreTraining


    class TFModel(keras.Model):

        def __init__(self, bert_embedding):
            super(TFModel, self).__init__()
            if not isinstance(bert_embedding, keras.Model):
                raise TypeError("TFModel only accept tf.keras.Model input!")
            self.bert_base = bert_embedding

        def model_save(self, path):
            self.trainable = True
            with Config.strategy.scope():
                self.save_weights(path)
            logger.info("Finish saving model to: '{}'.".format(path))

        def model_load(self, path):
            self.trainable = True
            with Config.strategy.scope():
                self.load_weights(path)
            logger.info("Finish loading model from: '{}'.".format(path))

        @abc.abstractmethod
        def call(self, inputs, training=None, mask=None):
            raise NotImplementedError

        @abc.abstractmethod
        def get_config(self):
            raise NotImplementedError

        def build_model(self, max_len):
            test_input = tf.convert_to_tensor(np.ones((1, max_len)).astype(np.int32))
            test_mask = tf.convert_to_tensor(np.ones((1, max_len)).astype(np.int32))
            test_typeid = tf.convert_to_tensor(np.ones((1, max_len)).astype(np.int32))

            inputs = test_input, test_mask, test_typeid
            self(inputs, training=False)


    def create_model(model_config):
        bert_config = BertConfig.from_pretrained(home_path() +
                                                 model_config["BASE_MODEL_PATH"], output_hidden_states=True,
                                                 output_attentions=True, return_dict=True)
        with Config.strategy.scope():
            bert_as_encoder = TFBertModel.from_pretrained(home_path() +
                                                          model_config["BASE_MODEL_PATH"], config=bert_config)
        return bert_as_encoder


    def create_pretraining_model(model_config):
        bert_config = BertConfig.from_pretrained(home_path() +
                                                 model_config["BASE_MODEL_PATH"], output_hidden_states=True,
                                                 output_attentions=True, return_dict=True)
        with Config.strategy.scope():
            bert_as_encoder = TFBertForPreTraining.from_pretrained(home_path() +
                                                          model_config["BASE_MODEL_PATH"], config=bert_config)
        return bert_as_encoder