import hypernlp.dl_framework_adaptor.models.pt_models as pt
import hypernlp.dl_framework_adaptor.models.tf_models as tf
from hypernlp.dl_framework_adaptor.configs.config import bert_models_config
from hypernlp.config import Config

from utils.string_utils import generate_model_name


def create_models(name, model_type):
    if Config.framework == "tensorflow":
        return tf.create_model(bert_models_config[generate_model_name(name, Config.framework, model_type)])
    elif Config.framework == "pytorch":
        return pt.create_model(bert_models_config[generate_model_name(name, Config.framework, model_type)])
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))


def create_pretraining_models(name, model_type):
    if Config.framework == "tensorflow":
        return tf.create_pretraining_model(bert_models_config[generate_model_name(name, Config.framework, model_type)])
    elif Config.framework == "pytorch":
        return pt.create_pretraining_model(bert_models_config[generate_model_name(name, Config.framework, model_type)])
    else:
        raise TypeError("Unsupported framework: '{}'".format(Config.framework))


if __name__ == '__main__':
    print(create_models(name="bert", model_type="chinese"))
