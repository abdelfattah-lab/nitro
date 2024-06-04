# Using this script
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

import numpy as np

import openvino.runtime.passes as passes

############## HELPER FUNCTIONS ##############

probe_enabled = True

def probe(model:ov.Model, *args):
    if not probe_enabled: return
    for a in args:
        print(f"{a}:", model.__getattribute__(a))

def visualize_example(m : ov.Model):
    # Need import:
    # * import openvino.runtime.passes as passes
    pass_manager = passes.Manager()
    pass_manager.register_pass(passes.VisualizeTree(file_name='image'))
    pass_manager.run_passes(m)

############## MODULES ##############

class SoftmaxModule(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxModule, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

class LinearModule(nn.Module):
  def __init__(self, in_features, out_features, bias=True):
    super(LinearModule, self).__init__()
    self.linear = nn.Linear(in_features, out_features, bias=bias)

  def forward(self, x):
    y = self.linear(x)
    return y

class Function(nn.Module):
  def __init__(self, in_features, out_features, bias=True, dim=-1):
    super(Function, self).__init__()
    self.linear = nn.Linear(in_features, out_features, bias=bias)
    self.softmax = SoftmaxModule(dim=dim)

  def forward(self, x):
    y = self.linear(x)
    y = self.softmax(y)
    return y

############## DRIVER CODE ##############

if __name__ == "__main__":
    # model = LinearModule(10, 2, bias=True)
    model = SoftmaxModule()
    # model = Function(in_features=10, out_features=2, bias=True, dim=-1)
    input = torch.randn(3, 10)
    memory = np.random.random((1, 10)).astype("float32") # to hold memory

    # --------- Using OpenVINO conversion --------- #

    converted_model = ov.convert_model(model, example_input=input) # why example input?

    probe(converted_model, "inputs")
    converted_model.reshape([1, 10]) # example_input was (1, 10), so this as well
    probe(converted_model, "inputs", "outputs")

    visualize_example(converted_model)
    
    # --------- Model Compilation --------- #

    compiled_model_gpu = ov.compile_model(converted_model, device_name="GPU")
    infer_request = compiled_model_gpu.create_infer_request()
    input_tensor = ov.Tensor(array=memory, shared_memory=True)

    infer_request.infer(input_tensor) # synchronous mode
    print(infer_request.get_input_tensor(0))

    # --------- Inference --------- #

    output_gpu = compiled_model_gpu(input)[0]
    probe(converted_model, "outputs")
    # print("GPU Output -\t", output_gpu, "\n")
