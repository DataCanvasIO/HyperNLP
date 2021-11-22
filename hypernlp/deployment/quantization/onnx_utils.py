import subprocess
import sys
import os
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)
import numpy as np

from hypernlp.framework_config import *
import torch.onnx
import tensorrt as trt


from utils.string_utils import home_path
from transformers.convert_graph_to_onnx import convert


def convert_data(model_path, model_name):

    if Config.framework == "tensorflow":
        framework = "tf"
    elif Config.framework == "pytorch":
        framework = "pt"
    else:
        raise ValueError("Unsupported framework: {}".format(Config.framework))
    convert(framework=framework, model=model_path,
            output=home_path() + 'hypernlp/deployment/onnx_models/' + model_name + '.onnx\'.', opset=11)
    print('Finish creating onnx model \'' +
          home_path() + 'hypernlp/deployment/onnx_models/' + model_name + '.onnx\'.')


def to_onnx_pt(model, model_name):
    torch.onnx.export(model, [torch.from_numpy(np.ones((1, 128)).astype(int)).cuda(), torch.from_numpy(
        np.ones((1, 128)).astype(int)).cuda(), torch.from_numpy(np.ones((1, 128)).astype(int)).cuda()],
                      home_path() + "hypernlp/deployment/onnx_models/" + model_name + ".onnx",
                      verbose=False, export_params=True, training=True, input_names=["inputs"],
                      output_names=["nsp", "mlm"], opset_version=12)
    print('Finish creating onnx model \'' +
                      home_path() + 'hypernlp/deployment/onnx_models/' + model_name + '.onnx\'.')


def to_onnx_tf(model, model_name):
    subprocess.Popen('python tf2onnx.convert --saved-model ' + model + ' --output ' +
                     home_path() + "hypernlp/deployment/onnx_models/" + model_name + '.onnx',
                     shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def convert2onnx(model, model_name):
    if Config.framework == "tensorflow":
        return to_onnx_tf(model, model_name)
    elif Config.framework == "pytorch":
        return to_onnx_pt(model, model_name)
    else:
        raise ValueError("Unsupported framework: {}".format(Config.framework))


if __name__ == '__main__':
    from hypernlp.nlp.task_models.pretraining import downstream_model
    import hypernlp.nlp.lm_models.bert as bert_model

    model, _ = downstream_model(128, bert_model.bert_model_chinese())
    # model_, _ = downstream_model(128, bert_model.bert_model_chinese(pretraining=False))

    # print(model, model_)

    model.model_load('/home/luhf/hypernlp/hypernlp/optimize/checkpoints/embedding_trained_model_.pth')

    to_onnx_pt(model.module, 'test_model')


