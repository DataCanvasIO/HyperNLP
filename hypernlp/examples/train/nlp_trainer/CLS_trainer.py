import os
import sys
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)
from hypernlp.framework_config import Config
from hypernlp.dl_framework_adaptor.configs.bertbase_config import bert_models_config
from hypernlp.dl_framework_adaptor.optimizer.optimizers import optimizer
from hypernlp.evaluation.evaluation_indicator import *
from hypernlp.evaluation.evaluator import PairWiseEvaluator
from hypernlp.nlp.data_process.reader import CSVReader, TXTReader
from hypernlp.nlp.dataset import DatasetSeq, DatasetSep
from hypernlp.nlp.task_models.cls import downstream_model
from hypernlp.nlp.tokenizer import TokenizerCLS
from hypernlp.nlp.tools.loss import focal_loss, ce_loss
from hypernlp.optimize.train_processor import train_processor
from hypernlp.optimize.trainer import trainer
from utils.gpu_status import environment_check
from utils.string_utils import generate_model_name, home_path
import hypernlp.nlp.lm_models.bert as bert_model


if __name__ == '__main__':
    environment_check()

    data = CSVReader(home_path() + "hypernlp/nlp/data/cls/", None)

    train_data_ = data.train_data(["content"], "class_label")
    validata_data_ = data.test_data(["content"], "class_label", True)

    cls_tokenizer = TokenizerCLS(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=256)

    eda = None#eda_model('cased', num_aug=2)

    train_data = DatasetSeq(train_data_, 256, cls_tokenizer, n_sampling=False,
                         batch_size=24, with_labels=True, EDA=eda)

    validata_data = DatasetSeq(validata_data_, 256, cls_tokenizer, n_sampling=False,
                            batch_size=24, with_labels=True, EDA=eda)

    model, _ = downstream_model(256, 3, bert_model.bert_model_chinese())

    # model.model_load('/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/0.h5')

    optimizer = optimizer(model, "adam",
                          {"lr": 0.00001, "betas": [0.9, 0.999], "epsilon": 1e-8, "amsgrad": False, "momentum": 0.9,
                           "weight_decay": 0.0005})

    evaluator = PairWiseEvaluator(validata_data, formulas=[acc_indicator, precision_indicator,
                                                           recall_indicator, f1_score_indicator])

    trainer = trainer(model, optimizer, train_processers=[train_processor(train_data, [ce_loss, 1.0])], epochs=4,
                      epoch_length=train_data.epoch_length,
                      save_model_path='/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/', evaluator=evaluator)
    trainer.train()
    model.model_save("/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/cls_trained_model_.h5")
