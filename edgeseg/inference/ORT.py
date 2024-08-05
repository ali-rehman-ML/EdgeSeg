import onnxruntime as onnxrt
import numpy as np


class ORT:
    def __init__(self, model):
        self.onnx_session= onnxrt.InferenceSession(model)
        self.onnx_inputs= self.onnx_session.get_inputs()[0].name

    def invoke(self,input):
        o = self.onnx_session.run(None, {self.onnx_inputs: input.astype(np.float32)})
        return o



