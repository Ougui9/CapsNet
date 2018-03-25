import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



# 1st Layer: convolutional layer: extracting features

class convReLU(nn.Module):
    '''
    in: (100, 1, 28, 28)
    out: (100, 256, 20, 20)
    '''
    def __init__(self, input_channel=1, output_channel=256, kernel=9,stride=1):
        super().__init__()
        self.conv=nn.Conv2d(input_channel,output_channel,kernel,stride)

    def forward(self, x):
        return F.relu((self.conv(x)))

