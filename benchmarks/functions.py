import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

import numpy as np

class Function(nn.Module):
    def __init__(self):
        super(Function, self).__init__()

    def forward(self, x):
        y = F.softmax(x, dim=1) # replace with any function
        return y

if __name__ == "__main__":
    torch.set_printoptions(precision=8)

    model = Function()
    input_tensor = torch.randn(1, 5)

    # Native inference in PyTorch
    print("Input  \t", input_tensor, "\n")
    output = model(input_tensor).numpy()
    print("Native Output -\t", output, "\n")

    # Using OpenVINO conversion
    converted_model = ov.convert_model(model,
                                       input=("x", ov.Shape([1, 5])),
                                       example_input=input_tensor
                                       )

    compiled_model_cpu = ov.compile_model(converted_model, device_name="CPU")
    compiled_model_gpu = ov.compile_model(converted_model, device_name="GPU")
    compiled_model_npu = ov.compile_model(converted_model, device_name="NPU")

    output_cpu = compiled_model_cpu(input_tensor)[0]
    output_gpu = compiled_model_gpu(input_tensor)[0]
    output_npu = compiled_model_npu(input_tensor)[0]
    
    # Inference
    print("CPU Output -\t", output_cpu)
    print("GPU Output -\t", output_gpu)
    print("NPU Output -\t", output_npu, "\n")

    print("Error CPU vs. Native - \t", np.mean(np.abs(output - output_cpu)))
    print("Error GPU vs. Native - \t", np.mean(np.abs(output - output_gpu)))
    print("Error NPU vs. Native - \t", np.mean(np.abs(output - output_npu)))
    
