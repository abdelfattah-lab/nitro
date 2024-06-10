import torch.nn as nn
import torch.nn.functional as F
import torch

# Just a bunch of models

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

class DualConvConcat(nn.Module):
  def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1, kernel_size2, stride1=1, stride2=1, padding1=0, padding2=0):
      super(DualConvConcat, self).__init__()
      self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size1, stride1, padding1)
      self.conv2 = nn.Conv2d(in_channels, out_channels2, kernel_size2, stride2, padding2)

  def forward(self, x, y):
      out1 = self.conv1(x)
      out2 = self.conv2(y)
      out = torch.cat((out1, out2), dim=1)
      return out
  
class Iterative(nn.Module):
  def __init__(self, features, bias=True, reps=5):
    super(Iterative, self).__init__()
    self.reps = reps
    for i in range(reps):
      linear_name = f"linear_{i+1}"
      setattr(self, linear_name, nn.Linear(features, features, bias=bias))
    
  def forward(self, x):
    for i in range(self.reps):
      linear_name = f"linear_{i+1}"
      layer = getattr(self, linear_name)
      x = layer(x)
    return x

class SmallConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel, stride, padding):
    super(SmallConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)

  def forward(self, x):
    return self.conv(x)