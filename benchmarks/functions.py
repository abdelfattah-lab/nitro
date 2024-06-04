# Testing PyTorch functions with OpenVINO, comparing CPU and GPU to test
# exact computation values.
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov

import numpy as np

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Example usage
if __name__ == "__main__":
    model = RMSNorm(5)
    input_tensor = torch.randn(1, 5, dtype=torch.float32)
    print(input_tensor)

    torch.set_printoptions(precision=8)
    print("Input  \t", input_tensor, "\n")
    output = model(input_tensor).detach().numpy()
    print("Native Output -\t", output, "\n")

    # Using OpenVINO conversion
    converted_model = ov.convert_model(model,
                                       input=("x", ov.Shape([1, 5]), ov.Type.f32),
                                       example_input=input_tensor
                                       )
    # converted_model.reshape([1, 10000])
    # print(converted_model.input("x"))
    # print(converted_model.outputs)
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
    
