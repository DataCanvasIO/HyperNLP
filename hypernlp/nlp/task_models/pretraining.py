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

        def __init__(self, bert_embedding, max_len):
            super(TFDSModel, self).__init__(bert_embedding)
            self.max_len = max_len

            self.__init_downstream_weights()

        def __init_downstream_weights(self):
            layer = self.bert_base.nsp.seq_relationship
            keras.initializers.glorot_normal(layer.weights[0])

            layer = self.bert_base.mlm.predictions
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

            embedding = self.bert_base(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       return_dict=True,
                                       output_hidden_states=True)

            seq_logits = embedding.seq_relationship_logits
            pred_logits = embedding.prediction_logits

            if training is False:
                seq_logits = nn.Softmax(dim=1)(seq_logits)
                pred_logits = nn.Softmax(dim=1)(pred_logits)

            return seq_logits, pred_logits

        def get_config(self):
            pass


elif Config.framework == 'pytorch':

    class PTDSModel(PTModelBase):
        '''
        pytorch downstream model
        '''

        def __init__(self, bert_embedding, max_len, gpu=True):
            super(PTDSModel, self).__init__(bert_embedding)
            self.max_len = max_len

        def __init_downstream_weights(self):
            for m in self.downstream_mlm.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # nn.init.xavier_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            for m in self.downstream_nsp.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # nn.init.xavier_normal_(m.weight)
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

        def forward(self, inputs, training=True):
            input_ids, attention_mask, token_type_ids = inputs

            self.modified_size(input_ids, attention_mask, token_type_ids)

            embedding = self.bert_base(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       return_dict=True,
                                       output_hidden_states=True)
            seq_logits = embedding.seq_relationship_logits
            pred_logits = embedding.prediction_logits
            pred_logits = torch.transpose(pred_logits, 1, 2)

            if training is False:
                seq_logits = nn.Softmax(dim=1)(seq_logits)
                pred_logits = nn.Softmax(dim=1)(pred_logits)

            # predict phase
            return seq_logits, pred_logits

        def train(self, mode=True):
            self.bert_base.train()

else:
    raise TypeError("Unsupported framework: '{}'".format(Config.framework))


def downstream_model(max_len, bert_base_model):

    if Config.framework == "tensorflow":
        # must load config file (.json) to allow BERT model return hidden states
        embedding = bert_base_model
        with Config.strategy.scope():
            model = TFDSModel(embedding, max_len)

        return model, tf_ce_loss

    elif Config.framework == "pytorch":
        embedding = bert_base_model
        model = PTDSModel(embedding, max_len)
        model = PTModel(model)
        model = model.cuda()

        return model, pt_ce_loss
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))


if __name__ == '__main__':

    import numpy as np
    import hypernlp.nlp.lm_models.bert as bert_model
    model, _ = downstream_model(128, bert_model.bert_model_chinese(pretraining=True))
    token = torch.from_numpy(np.ones((1, 128), np.int))
    mask = torch.from_numpy(np.zeros((1, 128), np.int))
    attantion = torch.from_numpy(np.zeros((1, 128), np.int))
    print(model((token, mask, attantion)))