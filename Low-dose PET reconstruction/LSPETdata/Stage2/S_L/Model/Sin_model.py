import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

'''
    Ordinary UNet Conv Block
'''
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation


        init.xavier_uniform_(self.conv.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv.bias,0)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0))
        init.constant_(self.conv2.bias,0)
    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out


'''    
 two-layer residual unit: two conv with BN/relu and identity mapping
'''
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.conv1.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        init.xavier_uniform_(self.conv2.weight, gain = np.sqrt(2.0)) #or gain=1
        init.constant_(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn2(self.conv2(out1)))
        bridge = x
        if self.in_size!=self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output
    
    

class ResNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=4):
        super(ResNet, self).__init__()
        #         self.imsize = imsize

        self.activation = F.relu


        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = residualUnit(64, 128)
        self.conv_block128_256 = residualUnit(128, 128)
        self.conv_block256_512 = residualUnit(128, 128)

        self.up_block512_256 = residualUnit(128, 128)
        self.up_block256_128 = residualUnit(128, 128)
        self.up_block128_64 = residualUnit(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1, stride=1)

    def forward(self, x):
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        block2 = self.conv_block64_128(block1)
        block3 = self.conv_block128_256(block2)
        block4 = self.conv_block256_512(block3)

        up2 = self.up_block512_256(block4)
        up3 = self.up_block256_128(up2)
        up4 = self.up_block128_64(up3)

        return self.last(up4)




























