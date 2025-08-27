import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        def CBR(i,o):
            return nn.Sequential(
                nn.Conv2d(i,o,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(o),
                nn.Conv2d(o,o,3,padding=1), nn.ReLU(True), nn.BatchNorm2d(o)
            )
        self.enc1 = CBR(in_ch, base)
        self.enc2 = CBR(base, base*2)
        self.enc3 = CBR(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bot  = CBR(base*4, base*8)
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = CBR(base*8 + base*4, base*4)
        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = CBR(base*4 + base*2, base*2)
        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = CBR(base*2 + base, base)
        self.out  = nn.Conv2d(base, out_ch, 1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

