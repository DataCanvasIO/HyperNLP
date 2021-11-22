import os
import sys

import numpy as np

root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)

from hypernlp.framework_config import Config
from hypernlp.nlp.task_models.pretraining import downstream_model
from hypernlp.nlp.dataset import DatasetSeq, DatasetCustom
from hypernlp.nlp.data_process.reader import CSVReader, TXTReader
from utils.string_utils import generate_model_name, home_path
from hypernlp.dl_framework_adaptor.configs.bertbase_config import bert_models_config
from hypernlp.nlp.tokenizer import TokenizerNSP, TokenizerCLS
import hypernlp.nlp.lm_models.bert as bert_model
from utils.gpu_status import environment_check
from hypernlp.evaluation.predictor import NLPPredictor


if __name__ == '__main__':

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
    IDX2CLS = {2: '负向', 1: '正向', 0: '中立'}

    data = CSVReader(home_path() + "hypernlp/nlp/data/nsp/", None)

    cls_tokenizer = TokenizerNSP(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=128)

    test_data_ = data.test_data(["s1", "s2"], "class_label", True)
    model, _ = downstream_model(128, bert_model.bert_model_cased())

    predictor = NLPPredictor(model)
    results = predictor.predict(data)

    with open(home_path() + 'hypernlp/examples/predict/nlp_predict/results/result_nsp.txt', 'w') as output:

        for d in range(len(test_data_)):
            cls = np.argmax(results[d])
            res = test_data_[d]
            output.write(res[0] + '\t' + str(cls) + '\t' + str(res[1]) + '\r')