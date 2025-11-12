import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList([
            nn.Linear(in_features=self.layer_sizes[i], out_features=self.layer_sizes[i+1]) for i in range(len(layer_sizes) -1)
        ])
        
    def forward(self, x):
        # no flatting necessary because dataset is already flat
        # in constrast to images which have px and channel
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class BinaryNeuralNet(nn.Module):
    def __init__(self):
        super(BinaryNeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=8)
        self.fc5 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        sig = nn.Sigmoid()
        return sig(x)
    
class BinaryNeuralNetSoftmax(nn.Module):
    def __init__(self):
        super(BinaryNeuralNetSoftmax, self).__init__()
        self.fc1 = nn.Linear(in_features=8, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=8)
        self.fc5 = nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        soft = F.log_softmax(x, dim=1)
        return soft