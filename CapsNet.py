from capsLayers import *


batchSize = 100
n_classes = 10
epochs = 500
rountingIteration = 3



class CapsNet(nn.Module):
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