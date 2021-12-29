import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.distributions as distributions
from easydict import EasyDict
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self, args, in_nc, nf):
        super(Encoder, self).__init__()
        self.conv1d = nn.Linear(in_features=in_nc, out_features=nf, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, input):
        # padding the one zero row to global position, so each joint including global position has 4 channels as input
        return self.lrelu(self.conv1d(input))

class Decoder(nn.Module):
    def __init__(self, args, nf, out_nc):
        super(Decoder, self).__init__()
        self.conv1d = nn.Linear(in_features=nf, out_features=out_nc, bias=True)

    def forward(self, input):
        return self.conv1d(input)

class Dense(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf=256, gc=256, dropout=0.25):
        super(Dense, self).__init__()
        self.linear_1 = nn.Linear(in_features=nf, out_features=gc, bias=True)
        self.linear_2 = nn.Linear(in_features=gc, out_features=nf, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, input):
        x = input
        x_f = self.lrelu(self.dropout(self.linear_1(input)))
        return self.lrelu(self.dropout(self.linear_2(x_f))) + x

class SimpleNet(nn.Module):
    """
    Temporal residual connected network for smoothing
    """
    def __init__(self, args, in_nc, out_nc, nf, nb, dropout, gc=256):
        super(SimpleNet, self).__init__()

        self.enc = Encoder(args, in_nc, nf)
#        out = make_uni_out(torch.ones((128,64,51)).cuda(), ratio=args.mask_rate, rand=True)
#        out = make_random_out(torch.ones((128,64,51)).cuda(), ratio=args.mask_rate)
#        self.enc = Encoder(args, out.shape[1], nf)
#        print('xxxx',int(in_nc*args.mask_rate),in_nc-int(in_nc*args.mask_rate),out.shape)
        self.dec = Decoder(args, nf, out_nc)
        self.args = args
        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(Dense(nf=nf, gc=gc, dropout=dropout))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

    def forward(self, x):
        N, T, K, C = x.size()
        x = x.reshape(N, T, K*C)

        x = x.permute(0,2,1) #[B, C, T]
        x_ = self.enc(x)
        x_ = self.ResidualBlock(x_)

        result = self.dec(x_)
        result = result.permute(0,2,1)
        result = result.reshape(N,T,K,-1)

        return result