from hypernlp.framework_config import Config
if Config.framework == 'tensorflow':
    import tensorflow.keras as keras
    from hypernlp.dl_framework_adaptor.models.tf_models import TFModel
elif Config.framework == 'pytorch':
    from hypernlp.dl_framework_adaptor.models.pt_models import PTModel, PTModelBase
else:
    raise TypeError("Unsupported framework: '{}'".format(Config.framework))
from hypernlp.nlp.tools.loss import *


if Config.framework == 'tensorflow':

    class TFDSModel(TFModel):
        '''
        tensorflow downstream model
        '''

        def __init__(self, bert_embedding, max_len, cls_num):
            super(TFDSModel, self).__init__(bert_embedding)
            self.max_len = max_len
            self.cls_num = cls_num

            self.downstream = keras.Sequential([
                keras.layers.Dense(64),
                keras.layers.Dropout(rate=0.3),
                keras.layers.ReLU(),
                keras.layers.Dense(self.cls_num)
            ])

        def init_downstream_weights(self):
            for layer in self.downstream.layers:
                if layer is isinstance(layer, keras.layers.Dense):
                    keras.initializers.glorot_normal(layer.weights[0])
                # elif layer is isinstance(layer, keras.layers.BatchNormalization):
                #     keras.initializers.ones(layer.weights[0])

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

        def call(self, inputs, training=False, mask=None):
            input_ids, attention_mask, token_type_ids = inputs
            input_ids = tf.reshape(input_ids, [-1, self.max_len])
            attention_mask = tf.reshape(attention_mask, [-1, self.max_len])
            token_type_ids = tf.reshape(token_type_ids, [-1, self.max_len])
            self.modified_size(input_ids, attention_mask, token_type_ids)

            if training is True:
                self.bert_base.layers[0].trainable = False
                self.bert_base.layers[0].pooler.trainable = True

            embedding = self.bert_base(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       return_dict=True,
                                       output_hidden_states=True)
            x = self.downstream(embedding.pooler_output)
            if training is not True:
                if self.cls_num == 1:
                    x = tf.nn.sigmoid(x)
                else:
                    x = tf.nn.softmax(x)
            return x

        def get_config(self):
            pass


elif Config.framework == 'pytorch':
    class PTDSModel(PTModel):
        '''
        pytorch downstream model
        '''

        def __init__(self, bert_embedding, max_len, cls_num, gpu=True):
            super(PTDSModel, self).__init__(bert_embedding)
            self.max_len = max_len
            self.cls_num = cls_num

            output_feature_size = 768

            self.downstream = nn.Sequential(
                nn.Flatten(),
                nn.Linear(output_feature_size, 64),
                nn.BatchNorm1d(64),
                # nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.cls_num))

            self.dropout_embed = nn.Dropout(self.dropout_emb)
            # self.dropout = nn.Dropout(self.dropout)

            self.bilstm = nn.LSTM(input_size=output_feature_size, hidden_size=self.cls_num, num_layers=2,
                                  bidirectional=True, batch_first=True, bias=True)

            self.linear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=self.cls_num, bias=True)

            if gpu:
                self.bert_base = self.bert_base.cuda()

                self.downstream = self.downstream.cuda()

        def init_downstream_weights(self):
            for m in self.downstream.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.xavier_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
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

        def forward(self, inputs):
            input_ids, attention_mask, token_type_ids = inputs

            self.modified_size(input_ids, attention_mask, token_type_ids)

            embedding = self.bert_base(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       return_dict=True,
                                       output_hidden_states=True)

            x = self.downstream(embedding.last_hidden_state)
            if self.cls_num == 1:
                x = nn.Sigmoid()(x)
            else:
                x = nn.Softmax(dim=1)(x)
            return x

        def train(self, mode=True):
            self.bert_base.encoder.train()
            self.bert_base.pooler.train()
            self.downstream.train()

else:
    raise TypeError("Unsupported framework: '{}'".format(Config.framework))

def downstream_model(max_len, cls_num, bert_base_model):

    if Config.framework == "tensorflow":
        # must load config file (.json) to allow BERT model return hidden states
        embedding = bert_base_model
        model = TFDSModel(embedding, max_len, cls_num)

        return model, tf_ce_loss

    elif Config.framework == "pytorch":

        embedding = bert_base_model
        model = PTDSModel(embedding, max_len, cls_num)

        return model, pt_ce_loss
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))


if __name__ == "__main__":

    from hypernlp.nlp.dataset import Dataset
    from hypernlp.nlp.data_process.reader import CSVReader
    from utils.string_utils import generate_model_name, home_path
    from hypernlp.dl_framework_adaptor.configs.bertbase_config import bert_models_config
    from utils.gpu_status import environment_check
    from hypernlp.nlp.tokenizer import TokenizerCLS

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}

    data = CSVReader("../data/", ["content"], CLS2IDX)

    cls_tokenizer = TokenizerCLS(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=128)

    data = Dataset(data.test_data, 128, tokenizer=cls_tokenizer, batch_size=12,
                   with_labels=False)

    # with tf.device('/cpu:0'):
    #     model, _ = downstream_model(128)
    #     pred = model(data.get_batch_data(), training=True)
    #     print(pred)

    model, _ = downstream_model(128, 3, bert_model.bert_model_cased())

    import time

    for i in range(10):
        start = time.time()
        model.train()
        pred = model(data.get_batch_data())
        print(time.time() - start, pred)
    # print(model)
