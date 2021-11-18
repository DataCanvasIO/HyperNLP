import os
import sys
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)
import torch


from hypernlp.config import Config
from hypernlp.nlp.task_models.cls import downstream_model
from hypernlp.nlp.dataset import DatasetSeq
from hypernlp.nlp.data_process.reader import CSVReader, TXTReader
from utils.string_utils import generate_model_name, home_path
from hypernlp.dl_framework_adaptor.configs.config import bert_models_config
from hypernlp.evaluation.evaluator import PairWiseEvaluator
from hypernlp.evaluation.evaluation_indicator import *
from hypernlp.nlp.tokenizer import TokenizerCLS
from utils.gpu_status import environment_check
import hypernlp.nlp.lm_models.bert as bert_model


if __name__ == '__main__':

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
    IDX2CLS = {2: '负向', 1: '正向', 0: '中立'}

    data = CSVReader(home_path() + "hypernlp/nlp/data/cls/", None)

    cls_tokenizer = TokenizerCLS(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=128)

    test_data_ = data.test_data(["content"], "class_label", True)

    validata_data = DatasetSeq(test_data_, 128, cls_tokenizer, n_sampling=False,
                               batch_size=48, with_labels=True)

    model, _ = downstream_model(128, 3, bert_model.bert_model_chinese())

    model.model_load('/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/cls_trained_model_.pth')

    evaluator = PairWiseEvaluator(validata_data,
                                  formulas=[acc_indicator, precision_indicator, recall_indicator,
                                            f1_score_indicator])

    print(evaluator.eval(model))
