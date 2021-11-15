import tensorflow.keras as keras

from hypernlp.nlp.tools.loss import *
from hypernlp.dl_framework_adaptor.models.pt_models import PTModel, PTModelBase
from hypernlp.dl_framework_adaptor.models.tf_models import TFModel
import hypernlp.nlp.lm_models.bert as bert_model


class TFDSModel(TFModel):
    '''
    tensorflow downstream model
    '''

    def __init__(self, bert_embedding, max_len, cls_num):
        super(TFDSModel, self).__init__(bert_embedding)
        self.max_len = max_len
        self.cls_num = cls_num

        self.bert_base.nsp.seq_relationship = tf.keras.layers.Dense(self.cls_num)

        self.bert_base.nsp.seq_relationship.build((None, 768))

        self.__init_downstream_weights()

    def __init_downstream_weights(self):
        layer = self.bert_base.nsp.seq_relationship
        keras.initializers.glorot_normal(layer.weights[0])

    def modified_size(self, input_ids, attention_mask, token_type_ids):
        input_size = input_ids.shape
        if not input_size[1] == self.max_len:
            raise ValueError(
                "input_ids length is out of range {} vs max_len: {}!".format(input_size[1], self.max_len))
        attention_mask = attention_mask.shape
        if not attention_mask[1] == self.max_len:
            raise ValueError(
                "attention_mask length is out of range {} vs max_len: {}!".format(attention_mask[1], self.max_len))
        token_type_ids = token_type_ids.shape
        if not token_type_ids[1] == self.max_len:
            raise ValueError(
                "token_type_ids length is out of range {} vs max_len: {}!".format(token_type_ids[1], self.max_len))

    def call(self, inputs, training=True, mask=None):
        input_ids, attention_mask, token_type_ids = inputs
        input_ids = tf.reshape(input_ids, [-1, self.max_len])
        attention_mask = tf.reshape(attention_mask, [-1, self.max_len])
        token_type_ids = tf.reshape(token_type_ids, [-1, self.max_len])
        self.modified_size(input_ids, attention_mask, token_type_ids)

        if training is True:
            self.bert_base.trainable = True
            self.bert_base.mlm.trainable = False

        embedding = self.bert_base(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   return_dict=True,
                                   output_hidden_states=True)
        x = embedding.seq_relationship_logits
        if training is False:
            if self.cls_num == 1:
                x = tf.nn.sigmoid(x)
            else:
                x = tf.nn.softmax(x)
        return x


class PTDSModel(PTModelBase):
    '''
    pytorch downstream model
    '''

    def __init__(self, bert_embedding, max_len, cls_num, gpu=True):
        super(PTDSModel, self).__init__(bert_embedding)
        self.max_len = max_len
        self.cls_num = cls_num

        output_feature_size = 768

        self.bert_base.cls.seq_relationship = nn.Linear(output_feature_size, self.cls_num)
        self.init_downstream_weights()

    def init_downstream_weights(self):
        m = self.bert_base.cls.seq_relationship
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    def modified_size(self, input_ids, attention_mask, token_type_ids):
        input_size = input_ids.shape
        if not input_size[1] == self.max_len:
            raise ValueError(
                "input_ids length is out of range {} vs max_len: {}!".format(input_size[1], self.max_len))
        attention_mask = attention_mask.shape
        if not attention_mask[1] == self.max_len:
            raise ValueError(
                "attention_mask length is out of range {} vs max_len: {}!".format(attention_mask[1], self.max_len))
        token_type_ids = token_type_ids.shape
        if not token_type_ids[1] == self.max_len:
            raise ValueError(
                "token_type_ids length is out of range {} vs max_len: {}!".format(token_type_ids[1], self.max_len))

    def forward(self, inputs, training=True):
        input_ids, attention_mask, token_type_ids = inputs

        self.modified_size(input_ids, attention_mask, token_type_ids)

        embedding = self.bert_base(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   return_dict=True,
                                   output_hidden_states=True)

        x = embedding.seq_relationship_logits
        if training is False:
            if self.cls_num == 1:
                x = nn.Sigmoid()(x)
            else:
                x = nn.Softmax(dim=1)(x)
        return x

    def train(self, mode=True):
        self.bert_base.train()


def downstream_model(max_len, cls_num, bert_base_model):

    if Config.framework == "tensorflow":
        # must load config file (.json) to allow BERT model return hidden states

        embedding = bert_base_model
        with Config.strategy.scope():
            model = TFDSModel(embedding, max_len, cls_num)
        return model, tf_ce_loss

    elif Config.framework == "pytorch":

        embedding = bert_base_model
        model = PTDSModel(embedding, max_len, cls_num)
        model = PTModel(model)
        model = model.cuda()

        return model, pt_ce_loss
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))


if __name__ == "__main__":

    from hypernlp.nlp.dataset import Dataset
    from hypernlp.nlp.data_process.reader import CSVReader
    from utils.string_utils import generate_model_name, home_path
    from hypernlp.dl_framework_adaptor.configs.config import bert_models_config
    from utils.gpu_status import environment_check
    from hypernlp.nlp.tokenizer import TokenizerCLS

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}

    data = CSVReader("../data/", ["content"], CLS2IDX)
    cls_tokenizer = TokenizerCLS(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=128)

    data = Dataset(data.test_data, 128, tokenizer=cls_tokenizer, batch_size=24,
                   with_labels=False)

    model, _ = downstream_model(128, 2, bert_model.bert_model_cased())

    print(model)

    import time

    for i in range(10):
        start = time.time()
        model.train()
        pred = model(data.get_batch_data())
        print(time.time() - start, pred)
    # print(model)
