import torch
import torch.nn as nn
import torch.nn.functional as F

class Indexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.randn([2])
    
    def forward(self, a:torch.Tensor, idx:torch.Tensor):
        idx = idx.squeeze().item()
        a[idx:idx+2] = torch.tensor([0, 0])
        return a

model = Indexer()

import openvino as ov
import openvino.runtime.passes as passes

TYPE = torch.float32

core = ov.Core()
ov_model = ov.convert_model(model,
                            example_input= [ torch.tensor([1,2,3,4,5], dtype=TYPE), torch.tensor([0]) ],
                            input=[[5], [1]])

# pass_manager = passes.Manager()
# pass_manager.register_pass(passes.VisualizeTree(file_name="image_2.svg"))
# pass_manager.run_passes(ov_model)


for device in ["CPU", "GPU", "NPU"]:
    model = ov.compile_model(ov_model, device_name=device)
    print(f"compiled on {device}")
    print(model([ torch.tensor([1,2,float("-inf"),4,5], dtype=TYPE), torch.tensor([3]) ]))