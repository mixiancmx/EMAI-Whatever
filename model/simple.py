#import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_embedding=nn.Conv1d(5,128,kernel_size=1)
        self.time_embedding=nn.Conv1d(4,1,kernel_size=1)
        self.FC=nn.Sequential(*[nn.Conv1d(128,128,kernel_size=1)]*5)
        self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.output=nn.Conv1d(128,1,kernel_size=1)


    def forward(self,x): # x [b,c,t]
        #print(x.shape)
        x=self.channel_embedding(x) 
        #print(x.shape)

        x=x.permute(0,2,1)
        #print(x.shape)
        x=self.time_embedding(x)
        #print(x.shape)
        x=x.permute(0,2,1)
        #print(x.shape)
        x=self.FC(x)
        x=self.relu(x)
        x=self.output(x)

        return x