import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        
        return out

class ResNet34Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNet34Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [BasicBlock(out_channels // 2 if downsample else out_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0 = self.maxpool(x0)
        
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return x1, x2, x3, x4

class UNetDecoder(nn.Module):
    def __init__(self, output_channels=1):
        super(UNetDecoder, self).__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)
    
    def forward(self, x1, x2, x3, x4):
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.conv1(x))
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv2(x))
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv3(x))
        
        x = self.up4(x)
        x1 = F.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=True) 
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv4(x))
        
        x = self.final_conv(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True) 
        return x

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34UNet, self).__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.decoder = UNetDecoder(out_channels)
    
    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4)
        return out


