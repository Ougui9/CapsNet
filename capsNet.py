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


class capsNetwork(nn.Module):
    def __init__(self):
        super().__init__()





# 1st Layer: convolutional layer: extracting features
class convLayer(nn.Module):
    def __init__(self, input_channel=1, output_channel=256, kernel=9,stride=1):
        super().__init__()
        self.conv=nn.Conv2d(input_channel,output_channel,kernel,stride)

    def forward(self, x):
        x=F.relu((self.conv(x)))
        return x


# 2st Layer: Primary Caps: generate capsules and activation by rounting
class primaryCaps(nn.Module):
    def __init__(self, n_capsules=8, input_channel=256, conv_kernel=9, conv_stride=2):
        super().__init__()
        try:
            output_channel = input_channel / n_capsules
        except ValueError:
            print('input_channel should be divisible by n_capsules')

        # output_channel=input_channel/n_capsules
        self.capsConv = nn.ModuleList([nn.Conv2d(input_channel, output_channel,conv_kernel,conv_stride) for i in range(10)])

    def forward(self, x):
        capsList=[caps(x) for caps in self.capsConv]



    def squashing(self):




# 3st Layer: Primary Caps
class digitCaps(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Module.__init__(self)
        self.squashing=primaryCaps.squashing




    def forward(self):













if __name__=='__main__':
    digitCaps()