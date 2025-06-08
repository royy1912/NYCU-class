import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        
        self.conv1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        #bottleneck
        self.conv5 = self.double_conv(512, 1024)

       
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = self.double_conv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = self.double_conv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = self.double_conv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = self.double_conv(128, 64)

       
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # Decoder
        u6 = self.up6(c5)
        c6 = self.conv6(torch.cat([u6, c4], dim=1))

        u7 = self.up7(c6)
        c7 = self.conv7(torch.cat([u7, c3], dim=1))

        u8 = self.up8(c7)
        c8 = self.conv8(torch.cat([u8, c2], dim=1))

        u9 = self.up9(c8)
        c9 = self.conv9(torch.cat([u9, c1], dim=1))

        output = self.final(c9)
        return output

    def double_conv(self, in_channels, out_channels):
        """ 兩層 3x3 Conv + ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )