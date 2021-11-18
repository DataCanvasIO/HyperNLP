import os
import sys

root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)
from hypernlp.config import Config
from hypernlp.nlp.task_models.pretraining import downstream_model
from hypernlp.nlp.dataset import DatasetSeq, DatasetSep, DatasetLM
from hypernlp.nlp.tools.loss import ce_loss
from hypernlp.nlp.data_process.reader import TXTReader, CSVReader
from utils.string_utils import generate_model_name, home_path
from hypernlp.dl_framework_adaptor.configs.config import bert_models_config
from hypernlp.dl_framework_adaptor.optimizer.optimizers import optimizer
from hypernlp.evaluation.evaluator import PairWiseEvaluator
from hypernlp.evaluation.evaluation_indicator import *
from hypernlp.nlp.tokenizer import TokenizerNSP
from hypernlp.optimize.trainer import trainer
from hypernlp.optimize.train_processer import train_processer
from utils.gpu_status import environment_check
import hypernlp.nlp.lm_models.bert as bert_model
from hypernlp.nlp.data_process.eda.eda import eda_model

import torch

if __name__ == '__main__':

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
    IDX2CLS = {2: '负向', 1: '正向', 0: '中立'}

    data = TXTReader(home_path() + "hypernlp/nlp/data/embedding/", None, spliter="###", skip_title=False)

    nsp_tokenizer = TokenizerNSP(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "cased")]["BASE_MODEL_PATH"], max_len=128)

    eda = None

    train_data_mlm = DatasetLM(data.train_data([0, 1], 2),
                               128, nsp_tokenizer, batch_size=16, with_labels=True, EDA=eda)

    train_data_nsp = DatasetSeq(data.train_data([0, 1], 2),
                                128, nsp_tokenizer, n_sampling=True, batch_size=16, with_labels=True, EDA=eda)

    model, _ = downstream_model(128, bert_model.bert_model_cased())

    # model.model_load('/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/embedding_trained_model_.pth')

    optimizer = optimizer(model, "adam",
                          {"lr": 0.00001, "betas": [0.9, 0.999], "epsilon": 1e-8, "amsgrad": False, "momentum": 0.9,
                           "weight_decay": 0.0005})
    # optimizer.load('/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/embedding_optimizer_cpt.pth')

    trainer = trainer(model, optimizer, train_processers=[train_processer(train_data_nsp, [ce_loss, 0.5]),
                                                          train_processer(train_data_mlm, [ce_loss, 0.5])], epochs=10,
                      epoch_length=train_data_mlm.epoch_length, evaluator=None)

    trainer.train()
    model.model_save("/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/embedding_trained_model_.pth")
    optimizer.save('/home/luhf/HyperNLP/hypernlp/optimize/checkpoints/embedding_optimizer_cpt.pth')
