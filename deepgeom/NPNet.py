#!/usr/bin/env python

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Functional

import time

from sklearn import manifold
from sklearn.utils import check_random_state

from deepgeom.pointnet import PointNetGlobalMax, get_MLP_layers, PointNetVanilla, PointNetVanilla1, PointwiseMLP

class myMSELoss(nn.Module):

    def foward(self, input1, input2):
        loss = ((input1 - input2)**2).mean()
        return loss


class NPNetSingle(nn.Module):
    def __init__(self, dims):
        super(NPNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class NPNetVanilla(nn.Module):             # PointNetVanilla or nn.Sequential
    def __init__(self, MLP_dims1, MLP_dims2 , MLP_dims21,
                 MLP_dims22):#, MLP_doLastRelu=False):
        assert(MLP_dims1[-1]==MLP_dims2[0])
        super(NPNetVanilla, self).__init__()

        self.PointNet1 = PointNetVanilla1(MLP_dims1, MLP_doLastRelu = True)
        self.PointNet2 = PointNetVanilla(MLP_dims2 , MLP_doLastRelu = False)
        self.N = 2048
        self.NPNet1 = PointNetVanilla1(MLP_dims21, MLP_doLastRelu = True)
        self.NPNet2 = PointNetVanilla1(MLP_dims22, MLP_doLastRelu = False)

    def forward(self, X):
        f = self.PointNet1.forward(X)
        local_feature = f

        f = self.PointNet2.forward(f)
        global_feature = (f.unsqueeze(1)).expand(-1 , self.N , -1)

        f = torch.cat((local_feature, global_feature), 2 )  # BxNx(128+512)
        f = self.NPNet1.forward(f)               # BxNx3 # B x N x 2 for isomap
        f = self.NPNet2.forward(f)               # BxNx3
        norm_f = (torch.norm(f[: , : , 0 : 3], p = 2, dim = 2 ).unsqueeze(2)).expand(-1 , -1 , 4)
        f1 = f/norm_f

        return f
