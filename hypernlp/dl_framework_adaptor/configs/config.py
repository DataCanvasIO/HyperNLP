from utils.string_utils import home_path
import yaml

configs = open(home_path() + "hypernlp/dl_framework_adaptor/configs/bert_config.yaml", encoding='utf-8')

bert_models_config = yaml.load(configs)

# print(bert_models_config)
