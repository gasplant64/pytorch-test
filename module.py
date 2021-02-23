### Pytorch module ###
import numpy as np

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as func
import math

class normal(nn.Module):
  def __init__(self, inputs , node ):
    super().__init__()
    self.weight = nn.Linear(inputs, node)
    self.bias   = nn.Parameter(torch.zeros(node)) 

  def forward(self, tmp):
    tmp = self.weight(tmp) + self.bias
    return tmp


loss = nn.CrossEntropyLoss()


if __name__ == '__main__':
  inp = torch.randn(4)
  test = normal(inputs = 4, node = 5)
  print(test(inp))
