import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np


batchSize = 100
n_classes = 10
epochs = 500
rountingIteration = 3

def squash(x):
    squared_norm = (x ** 2).sum(-1, keepdim=True)
    return squared_norm/(1+squared_norm)*x/torch.sqrt(squared_norm)







# 1st Layer: convolutional layer: extracting features
class convReLU(nn.Module):
    def __init__(self, input_channel=1, output_channel=256, kernel=9,stride=1):
        super().__init__()
        self.conv=nn.Conv2d(input_channel,output_channel,kernel,stride)

    def forward(self, x):
        x=F.relu((self.conv(x)))
        return x


# 2st Layer: Primary Caps: generate capsules and activation by rounting
class primaryCaps(nn.Module):
    def __init__(self, n_capsules=8,n_routingNode=6*6*32,n_iteration=3, input_channel=256, conv_kernel=9, conv_stride=2):
        super().__init__()
        self.in_channels=input_channel
        self.conv_stride=conv_stride
        self.n_capsules=n_capsules
        self.n_routingNode=n_routingNode
        self.n_iteration=n_iteration
        try:
            output_channel = input_channel / n_capsules
        except ValueError:
            print('input_channel should be divisible by n_capsules')

        # output_channel=input_channel/n_capsules
        self.capsConv = nn.ModuleList([nn.Conv2d(input_channel, output_channel,conv_kernel,conv_stride) for _ in range(10)])
        self.w=nn.Parameter(torch.rand(n_capsules, n_routingNode, input_channel, output_channel))

    def forward(self, x):
        # capsList=[caps(x) for caps in self.capsConv]
        caps_out =[]
        for capsconv in self.capsConv:
            caps_out.append(capsconv(x))
        caps_out=torch.stack(caps_out,dim=1)

        # caps_out=torch.cat(caps_out,dim=1)#(256,6*6*32
        vector=caps_out.view(batchSize,self.n_routingNode,-1)#(100,6*6*32)
        vector=self.squashing(vector)
        x = torch.stack([vector] * self.n_capsules, dim=2)[:,:,:,:,None]



        return vector


    def squashing(self,xx):
        return squash(xx)



# 3st Layer: Primary Caps
class digitCaps(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Module.__init__(self)
        self.squashing=primaryCaps.squashing




    def forward(self):

class decoder(nn.Module):
    def __init__(self):
        super().__init__()




class Caps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=convReLU
        self.primary_caps=primaryCaps
        self.digit_caps=digitCaps


    def forward(self, x):
        # out=nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9)
        out=self.conv(x)
        out=self.primary_caps(out)
        out=self,digitCaps(out)









if __name__=='__main__':
    digitCaps()