import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import torch
import torch.nn as nn
"""noise = np.random.normal(...)


for param1, param2 in product():
    ...
"""
input = torch.randn(2, 2, requires_grad=True)
target = torch.tensor([1, 0])
log_softmax = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()
print(log_softmax(input))
loss = loss_fn(log_softmax(input), target)
