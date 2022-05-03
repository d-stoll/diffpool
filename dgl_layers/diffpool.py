import torch
from dgl.nn.pytorch import DenseSAGEConv
from torch import nn as nn
from torch.nn import BatchNorm1d

from .assignment import DiffPoolAssignment
from .loss import EntropyLoss, LinkPredLoss


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.link_pred_layer = LinkPredLoss()
        self.embed = DenseSAGEConv(nfeat, nhid, norm=BatchNorm1d(nhid))
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log['a'] = anext.cpu().numpy()
        return xnext, anext