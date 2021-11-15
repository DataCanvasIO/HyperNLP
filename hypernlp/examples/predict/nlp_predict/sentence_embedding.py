import os
import sys
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)
import torch


from hypernlp.config import Config
from hypernlp.nlp.task_models.pretraining import downstream_model
from hypernlp.nlp.dataset import DatasetSeq, DatasetCustom
from hypernlp.nlp.data_process.reader import CSVReader, TXTReader
from utils.string_utils import generate_model_name, home_path
from hypernlp.dl_framework_adaptor.configs.config import bert_models_config
from hypernlp.nlp.tokenizer import TokenizerNSP, TokenizerCLS
import hypernlp.nlp.lm_models.bert as bert_model
from utils.gpu_status import environment_check


if __name__ == '__main__':

    environment_check()

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
    IDX2CLS = {2: '负向', 1: '正向', 0: '中立'}

    # data = TXTReader("/home/luhf/compatition/", [0, 1, 2], None, label_index=3, spliter="###", skip_title=False)
    data = CSVReader("/home/luhf/compatition/", ['seq', 'line_id', 'file_id'], None)

    nsp_tokenizer = TokenizerCLS(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=128)

    data = DatasetCustom(data.test_data, 128, nsp_tokenizer,
                         batch_size=200, data_column=[[2], [0, 1]], shuffle=True)

    model, _ = downstream_model(128, bert_model.bert_model_cased())

    with open('./result_embedding.txt', 'w') as output:
        for i in range(data.epoch_length):
            batch_data = data.get_batch_data()
            if Config.framework == "tensorflow":
                pred = model(batch_data[:3], training=False).numpy()
                tail = batch_data[-1]
            elif Config.framework == "pytorch":
                model.eval()
                with torch.no_grad():
                    pred = model(batch_data[:3]).data.cpu().numpy()
                    tail = batch_data[-1]
            else:
                raise ValueError("Unsupported framework: {}".format(Config.framework))
            for d in range(data.batch_size):
                # cls = 1 if pred[d] >= 0.5 else 0
                output.write(str(tail[d][0]) + "\t" + str(tail[d][1]) + "\t" +
                             str(list(pred[d])).replace('[', '').replace(']', '') + "\n")
            sys.stdout.write("Processing evaluation: {}/{}".format(i, data.epoch_length) + '\r')
            sys.stdout.flush()



