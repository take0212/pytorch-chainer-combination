import torch
from torch import nn
import torch.nn.functional as F
#----
import pytorch_pfn_extras as ppe

class MLP(nn.Module):

    # def __init__(self, n_mid_units=100, n_out=10):
    def __init__(self, n_mid_units=100, n_out=10, lazy=True):
        super(MLP, self).__init__()
        if lazy:
            self.l1 = ppe.nn.LazyLinear(None, n_mid_units)
            self.l2 = ppe.nn.LazyLinear(None, n_mid_units)
            self.l3 = ppe.nn.LazyLinear(None, n_out)
        else:
            self.l1 = nn.Linear(784, n_mid_units)
            self.l2 = nn.Linear(n_mid_units, n_mid_units)
            self.l3 = nn.Linear(n_mid_units, n_out)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return F.log_softmax(h3, dim=1)

