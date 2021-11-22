import sys
import os
root_path = os.path.join(os.getcwd(), "")
sys.path.append(root_path)

import onnx

from utils.string_utils import home_path

max_batch_size = 1

test = onnx.load(home_path() + "hypernlp/deployment/onnx_models/test_model.onnx")
onnx.checker.check_model(test)
print("==> Passed")