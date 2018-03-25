import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np

def squash(x):
    squared_norm = (x ** 2).sum(-1, keepdim=True)
    return squared_norm/(1+squared_norm)*x/torch.sqrt(squared_norm)





# 2st Layer: Primary Caps: generate capsules
class primaryCaps(nn.Module):
    '''
    input: (batchsize, )(100, 256,20,20)
    out: (100, 1152, 8)
    '''
    def __init__(self, pixel_each_capsule=8, input_channel=256, conv_kernel=9, conv_stride=2):
        super().__init__()
        self.in_channels=input_channel
        self.conv_stride=conv_stride
        self.pixel_each_capsule=pixel_each_capsule
        # self.n_routingNode=n_routingNode
        # self.n_iteration=n_iteration
        try:
            output_channel = input_channel / pixel_each_capsule
        except ValueError:
            print('input_channel should be divisible by n_capsules')

        # output_channel=input_channel/n_capsules
        self.capsConv = nn.ModuleList([nn.Conv2d(input_channel, output_channel,conv_kernel,conv_stride) for _ in range(10)])


    def forward(self, x):
        # capsList=[caps(x) for caps in self.capsConv]
        caps_out =[]
        for capsconv in self.capsConv:
            caps_out.append(capsconv(x))
        caps_out=torch.stack(caps_out,dim=1)

        # caps_out=torch.cat(caps_out,dim=1)#(256,6*6*32
        vector=caps_out.view(x.size(0),self.n_routingNode,-1)#(100,6*6*32)
        vector=self.squashing(vector)
        # x = torch.stack([vector] * self.n_capsules, dim=2)[:,:,:,:,None]



        return vector


    def squashing(self,xx):
        return squash(xx)



# 3st Layer: Digit Caps
# activation by rounting
class digitCaps(nn.Module):
    '''
    in: (100, 1152, 8)
    out: (100, 10, 16, 1)
    '''
    def __init__(self,n_capsules=10, n_routingNode=6*6*32, in_channels=8, out_channels=16, n_iteration=3):
        super().__init__()
        # nn.Module.__init__(self)
        self.squashing=squash

        self.n_routingNode = n_routingNode
        self.n_iteration = n_iteration


        self.w = nn.Parameter(torch.rand(n_capsules, n_routingNode, in_channels, out_channels))







    def forward(self):

        return


    def squashing(self,xx):
        return squash(xx)


class decoder(nn.Module):
    '''
    in: (100, 10, 16, 1)
    out:
        output: (100, 10, 16, 1)
        recon: (100, 1, 28, 28)
        masked: (100,10)
    '''
    def __init__(self):
        super().__init__()














if __name__=='__main__':
    digitCaps()