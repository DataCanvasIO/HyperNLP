import os
import sys
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)
from hypernlp.framework_config import Config
from hypernlp.nlp.task_models.nsp import downstream_model
from hypernlp.nlp.dataset import DatasetSeq, DatasetSep
from hypernlp.nlp.tools.loss import ce_loss
from hypernlp.nlp.data_process.reader import TXTReader, CSVReader
from utils.string_utils import generate_model_name, home_path
from hypernlp.dl_framework_adaptor.configs.bertbase_config import bert_models_config
from hypernlp.dl_framework_adaptor.optimizer.optimizers import optimizer
from hypernlp.evaluation.evaluator import PairWiseEvaluator
from hypernlp.evaluation.evaluation_indicator import *
from hypernlp.nlp.tokenizer import TokenizerNSP
from hypernlp.optimize.train_processor import train_processor
from hypernlp.optimize.trainer import trainer
from utils.gpu_status import environment_check
import hypernlp.nlp.lm_models.bert as bert_model
from hypernlp.nlp.data_process.eda.eda import eda_model


if __name__ == '__main__':

    environment_check()

    data = CSVReader(home_path() + "hypernlp/nlp/data/nsp/", None)

    train_data_ = data.train_data(["s1", "s2"], "class_label")
    validata_data_ = data.validate_data(["s1", "s2"], "class_label")

    nsp_tokenizer = TokenizerNSP(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=196)

    eda = None  # eda_model('cased', num_aug=2)

    train_data = DatasetSeq(train_data_, 196, nsp_tokenizer, n_sampling=False,
                            batch_size=48, with_labels=True, EDA=eda)

    validata_data = DatasetSeq(validata_data_, 196, nsp_tokenizer, n_sampling=False,
                               batch_size=48, with_labels=True, EDA=eda)

    model, _ = downstream_model(128, 1, bert_model.bert_model_chinese())

    optimizer = optimizer(model, "adam",
                          {"lr": 0.00001, "betas": [0.9, 0.999], "epsilon": 1e-8, "amsgrad": False, "momentum": 0.9,
                           "weight_decay": 0.0005})

    evaluator = PairWiseEvaluator(validata_data, formulas=[acc_indicator, precision_indicator,
                                                           recall_indicator, f1_score_indicator])

    trainer = trainer(model, optimizer, train_processers=[train_processor(train_data, [ce_loss, 1.0])], epochs=4,
                      epoch_length=train_data.epoch_length,
                      save_model_path='/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/', evaluator=evaluator)
    trainer.train()
    model.model_save("/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/nsp_trained_model_.pth")
