from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import tqdm
import logging
import math
from os import path
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils import data
from torch.utils.data import DataLoader
# from torch_geometric.nn import GCNConv, GATConv

import time
from model.layers import *



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super(GCN, self).__init__()

        self.gc1 = GCNLayer(nfeat, args.nhid)
        self.gc2 = GCNLayer(args.nhid, args.nhid)
        self.dropout = args.drop

    def forward(self, x, adj):
        x = self.gc1(x, adj)        
        x = F.normalize(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc2(x, adj)
        x = F.normalize(x)
        #x = F.dropout(x, self.dropout, training=self.training)
        return x 


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super(GAT, self).__init__()

        alpha=0.2
        nheads=3
        self.dropout = args.drop

        self.attentions = [SpGraphAttentionLayer(nfeat, args.nhid, dropout=self.dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(args.nhid * nheads, nclass, dropout=self.dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # print(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super(GraphSAGE, self).__init__()

        self.sage1 = SAGELayer(nfeat, args.nhid)
        self.sage2 = SAGELayer(args.nhid*2, args.nhid)
        self.fc = nn.Linear(args.nhid*2, nclass, bias=True)
        self.dropout = args.drop

    def forward(self, x, adj):

        x = F.relu(self.sage1(x, adj))
        x = F.normalize(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.normalize(x)

        x = self.fc(x)
        return x #F.log_softmax(x, dim=1)
