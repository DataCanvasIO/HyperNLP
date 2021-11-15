import tensorrt

from hypernlp.deployment.quantization.onnx_utils import ONNX_build_engine


print(tensorrt.__version__)

def load_onnx_model(model_path):
    pass


def create_tensorrt_inference_from_engine(engine):

    engine = ONNX_build_engine(filepath)

    context = engine.create_execution_context()

    # 分配内存
    d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
    d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    bindings = [int(d_input), int(d_output)]

    # pycuda操作缓冲区
    stream = cuda.Stream()
    # 将输入数据放入device
    cuda.memcpy_htod_async(d_input, img, stream)
    # 执行模型
    context.execute_async(100, bindings, stream.handle, None)
    # 将预测结果从从缓冲区取出
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # 线程同步
    stream.synchronize()

    print("Test Case: " + str(target))
    print("Prediction: " + str(np.argmax(output, axis=1)))
    print("tensorrt time:", time() - Start)

    del context
    del engine