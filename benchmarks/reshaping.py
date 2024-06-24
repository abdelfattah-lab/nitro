import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import time

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DoubleFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: int,
    ):
        super(DoubleFeedForward, self).__init__()
        self.ffn1 = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
        self.ffn2 = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.ffn2(x)
        return x

# PARAMETERS
dim = 512  # Dimension of the input
hidden_dim = 2048  # Hidden dimension before adjustments
multiple_of = 256  # Multiple for adjusting hidden dimension
ffn_dim_multiplier = 4  # Custom multiplier for hidden dimension

ffn = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)

batch_size = 32
seq_length = 10
input_tensor = torch.rand(seq_length, batch_size, dim)  # Sequence length 10, batch size 32, feature size dim

import openvino as ov
import openvino.runtime.passes as passes

ov_model = ov.convert_model(ffn, example_input=input_tensor)
ov_model.reshape([(seq_length, seq_length+1), batch_size, dim])
utils.visualize_example(ov_model, "double_ffn_layer")

core = ov.Core()
# Dynamic
for device in ["NPU"]:

    # Compilation
    start = time.time()
    model = core.compile_model(ov_model, device_name=device)
    print(f"{device} Compile time: {time.time() - start}")

    # Inference, 1000 times
    start = time.time()
    for _ in range(1000):
        input = torch.rand(seq_length, batch_size, dim)
        output = model(input)
    print(f"{device} Inference time: {time.time() - start}")


