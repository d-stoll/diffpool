from dgl.nn.pytorch import DenseSAGEConv

from torch import nn as nn
from torch.nn import functional as F, BatchNorm1d


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = DenseSAGEConv(nfeat, nnext, norm=BatchNorm1d(nnext))

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l