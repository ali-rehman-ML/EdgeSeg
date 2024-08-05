import tflite_runtime.interpreter as tflite
import numpy as np


class TFRT:
    def __init__(self, model):
        self.interpreter = tflite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def invoke(self,input):
        self.interpreter.set_tensor(self.input_details[0]['index'], input.astype(np.float32))
        self.interpreter.invoke()
        o=self.interpreter.get_tensor(self.output_details[0]['index'])
        return o



