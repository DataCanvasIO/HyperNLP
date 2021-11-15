from hypernlp.dl_framework_adaptor.models.pretrained_models import create_models, create_pretraining_models

bert_seq_embedding_size = {'chinese': 21128, 'cased': 28996}


def bert_model_chinese(pretraining=True):
    # must load config file (.json) to allow BERT model return hidden states
    if pretraining is False:
        bert_base = create_models(name="bert", model_type="chinese")
    else:
        bert_base = create_pretraining_models(name="bert", model_type="chinese")
    return bert_base


def bert_model_cased(pretraining=True):
    # must load config file (.json) to allow BERT model return hidden states
    if pretraining is False:
        bert_base = create_models(name="bert", model_type="cased")
    else:
        bert_base = create_pretraining_models(name="bert", model_type="cased")
    return bert_base


