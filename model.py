import torch 
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualBlock, self).__init__()
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


    def forward(self, x):
        x_out = self.conv2(self.conv1(x))

        if self.is_res:
            

        return self.relu(out)

class DownwardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownwardBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.res_block(x)
        return self.pool(x), x

class UpwardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpwardBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upsample(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.res_block(x)
