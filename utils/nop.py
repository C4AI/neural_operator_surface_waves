import torch
import numpy as np
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from utils.nn_conv import NNConv_old
import torch.nn.functional
from scipy.io import loadmat
import random
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KernelNN(torch.nn.Module):

#This class was adapted from https://github.com/zongyi-li/graph-pde
#Z. Li et al., “Neural Operator: Graph Kernel Network for Partial Differential Equations,” arXiv:2003.03485 [cs, math, stat], Mar. 2020, Accessed: Apr. 19, 2022. [Online]. Available: http://arxiv.org/abs/2003.03485

    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel = DenseNet([ker_in, width_kernel, width_kernel, width_node**2], torch.nn.ReLU, normalize=True)
        self.conv1 = NNConv_old(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, out_width)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = torch.nn.functional.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class DenseNet(torch.nn.Module):

#This class was cloned/adapted from https://github.com/zongyi-li/graph-pde
#Z. Li et al., “Neural Operator: Graph Kernel Network for Partial Differential Equations,” arXiv:2003.03485 [cs, math, stat], Mar. 2020, Accessed: Apr. 19, 2022. [Online]. Available: http://arxiv.org/abs/2003.03485

    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x