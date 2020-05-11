# import
import torch
import torch.nn.functional as F
from torch import nn

class MLP(nn.Module):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, n_mid_units)
        self.l2 = nn.Linear(n_mid_units, n_mid_units)
        self.l3 = nn.Linear(n_mid_units, n_out)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return F.log_softmax(h3, dim=1)

