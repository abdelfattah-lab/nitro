import torch.nn as nn
import torch.nn.functional as F
import torch
import openvino as ov
import time

from models import Iterative, SmallConv
from utils import load_or_convert, visualize_example

VISUALIZE = False

# Sequential parameters
features=64
reps=500

# Convolutional parameters
size=64
in_channels=256
out_channels=256
kernel=3
stride=1
padding=1

class SeqConv(nn.Module):
    def __init__(self, features, reps, in_channels, out_channels, kernel, stride, padding):
        super(SeqConv, self).__init__()
        self.model_seq = Iterative(features=features, reps=reps)
        self.model_conv = SmallConv(in_channels=in_channels, out_channels=out_channels, kernel=kernel, stride=stride, padding=padding)
    
    def forward(self, x, y):
        output_x = self.model_seq(x)
        output_y = self.model_conv(y)
        
        return output_x + output_y

# ------- Model conversion/loading -------

model = SeqConv(features, reps, in_channels, out_channels, kernel, stride, padding)
core = ov.Core()
ov_model = load_or_convert("seqconv_module", model, force_update=False, core=core,
                           example_input=[torch.randn(1,features), torch.randn(1,in_channels,size,size)],
                           input=[
                                ("x", [1, features]),
                                ("y", [1,in_channels,size,size])
                                ]
                            )
if VISUALIZE:
    visualize_example(ov_model, "seqconv_module")

# ------- Compilation -------

device = "HETERO:CPU,GPU"

convolution = {
    "Constant_6006",
    "Parameter_4",
    "Convert_6008",
    "Constant_6012",
    "Convolution_6010",
    "Convert_6014",
    "Add_6016",
}
end = {
    "Add_6018",
    "Result_6020"
}
# everything else is in the sequential

# adjust affinity for compilation
for node in ov_model.get_ops():
        name = node.get_name()
        if name not in convolution and name not in end:     # SEQUENTIAL
            node.get_rt_info()["affinity"] = "NPU"
        elif name in convolution:                           # CONVOLUTIONAL
            node.get_rt_info()["affinity"] = "GPU"
        elif name in end:                                   # END ADDITION AND RESULT
            node.get_rt_info()["affinity"] = "CPU"
        else:
            raise Exception("node not found!")

for device in ["HETERO:CPU,GPU,NPU"]:
    start = time.time()
    compiled_model = ov.compile_model(model=ov_model, device_name=device)
    print(f"{device} Compiling time: {time.time() - start}")
    start = time.time()
    for _ in range(1000):
        input1 = torch.randn(1, features)
        input2 = torch.randn(1, in_channels, size, size) 
        output = compiled_model((input1, input2))
    print(f"{device} Inference time: {time.time() - start}")
    print()
