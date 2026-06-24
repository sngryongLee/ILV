import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """ 3D residual block """

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(4, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(x + y)


class Refine3D(nn.Module):
    """
    3D volume refinement module
    """
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(8, 16, 4, stride=4),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True)
        )
        self.down1_res = ResBlock3D(16)

        self.down2 = nn.Sequential(
            nn.Conv3d(16, 32, 2, stride=2),
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=True)
        )

        self.body = nn.Sequential(
            ResBlock3D(32),
            ResBlock3D(32)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True)
        )
        self.up2_res = ResBlock3D(16)

        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 4, stride=4),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv3d(8, 1, 1)

    def forward(self, vol):
        x1 = self.head(vol)

        x2 = self.down1(x1)
        x2 = self.down1_res(x2)

        x3 = self.down2(x2)

        y = self.body(x3)

        y = self.up2(y)
        y = y + x2
        y = self.up2_res(y)

        y = self.up1(y)
        y = y + x1

        delta = self.tail(y)
        return vol + delta



class Refine3DLegacy(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )

        self.down = nn.Sequential(
            nn.Conv3d(8, 16, 4, stride=4, padding=0),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            ResBlock3D(16),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 4, stride=4, padding=0),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv3d(8, 1, 1)

    def forward(self, v):
        y = self.head(v)
        y = self.down(y)
        y = self.body(y)
        y = self.up(y)
        delta = self.tail(y)
        return v + delta
