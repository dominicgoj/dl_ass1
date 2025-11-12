import torch
import numpy as np

array = [[1, 2], [3, 4], [5, 6]]

np_array = np.array(array)
torch_tensor = torch.tensor(array)
tt_from_np = torch.from_numpy(np_array)
print(torch_tensor.shape)

ones = torch.ones((3, 4))
zeros = torch.zeros((2, 5))

arange = torch_tensor.reshape(2,3)
print(arange)