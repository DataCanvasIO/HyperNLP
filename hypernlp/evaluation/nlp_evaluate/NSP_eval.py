import os
import sys
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)

from hypernlp.framework_config import Config
from hypernlp.nlp.task_models.nsp import downstream_model
from hypernlp.nlp.dataset import DatasetSeq
from hypernlp.nlp.data_process.reader import CSVReader, TXTReader
from utils.string_utils import generate_model_name, home_path
from hypernlp.dl_framework_adaptor.configs.bertbase_config import bert_models_config
from hypernlp.evaluation.evaluator import PairWiseEvaluator
from hypernlp.evaluation.evaluation_indicator import *
from hypernlp.nlp.tokenizer import TokenizerNSP
from utils.gpu_status import environment_check
import hypernlp.nlp.lm_models.bert as bert_model


if __name__ == '__main__':

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
    IDX2CLS = {2: '负向', 1: '正向', 0: '中立'}

    data = CSVReader(home_path() + "hypernlp/nlp/data/nsp/", None)

    nsp_tokenizer = TokenizerNSP(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "cased")]["BASE_MODEL_PATH"], max_len=128)

    test_data = DatasetSeq(data.test_data, 128, nsp_tokenizer, n_sampling=False,
                           batch_size=64, with_labels=True)

    model, _ = downstream_model(128, 1, bert_model.bert_model_cased())

    model.load_state_dict('/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/14.pth')

    evaluator = PairWiseEvaluator(test_data,
                                  formulas=[acc_indicator, precision_indicator, recall_indicator,
                                            f1_score_indicator])

    print(evaluator.eval(model))
