import torch
import torch.nn as nn


class WeightConv(nn.Module):
    def __init__(self, nf=64, scale=2, out_chls=[24, 3]):
        super(WeightConv, self).__init__()

        self.up_conv = nn.Conv2d(nf, nf * scale ** 2, kernel_size=1, stride=1, padding=0)
        self.ps = nn.PixelShuffle(upscale_factor=scale)
        self.relu = nn.ReLU(inplace=True)

        self.angle_conv = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=out_chls[0], kernel_size=1, stride=1, padding=0)
        )

        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=out_chls[1], kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.relu(self.ps(self.up_conv(x)))

        return self.angle_conv(x), self.scale_conv(x),x


class WeightConvKeep(nn.Module):
    def __init__(self, in_chl=3, nf=64, out_chls=[24, 3]):
        super(WeightConvKeep, self).__init__()

        self.conv1 = nn.Conv2d(in_chl, nf, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.angle_conv = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=out_chls[0], kernel_size=1, stride=1, padding=0)
        )

        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=out_chls[1], kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))

        return self.angle_conv(x), self.scale_conv(x)


if __name__ == '__main__':
    conv = WeightConv()
    input = torch.randn(1, 64, 4, 4)
    angle, scale = conv(input)
    print(angle.size(), scale.size())