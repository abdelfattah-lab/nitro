# Just a bunch of models
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxModule(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxModule, self).__init__()
        self.dim = dim

    def forward(self, x):
        y = F.softmax(x, dim=self.dim)
        return y

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
    self.linear = LinearModule(in_features, out_features, bias=bias)
    self.softmax = SoftmaxModule(dim=dim)

  def forward(self, x):
    y = self.linear(x)
    y = self.softmax(y)
    return y