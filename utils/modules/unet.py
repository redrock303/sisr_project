import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())

    return nn.Sequential(*layers)


class block(nn.Module):
    def __init__(self, nf=64):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        out = self.relu(x + identity)

        return out


class UNet_down8(nn.Module):
    def __init__(self, nf=64):
        super(UNet_down8, self).__init__()
        # downsample
        self.conv2_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv4_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv4_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # skip
        self.skip1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.skip2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.skip3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.skip4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsample
        self.deconv3 = nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.deconv1 = nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        L1 = x
        L2 = self.relu(self.conv2_2(self.relu(self.conv2_1(L1))))
        L3 = self.relu(self.conv3_2(self.relu(self.conv3_1(L2))))
        L4 = self.relu(self.conv4_2(self.relu(self.conv4_1(L3))))

        # L4
        L4_out = self.relu(self.skip4(L4))
        # L3
        L3_skip = self.relu(self.skip3(L3))
        L4_up = self.relu(self.deconv3(L4_out))
        L3_cat = torch.cat([L3_skip, L4_up], dim=1)
        L3_out = self.relu(self.upconv3(L3_cat))
        # L2
        L2_skip = self.relu(self.skip2(L2))
        L3_up = self.relu(self.deconv2(L3_out))
        L2_cat = torch.cat([L2_skip, L3_up], dim=1)
        L2_out = self.relu(self.upconv2(L2_cat))
        # L1
        L1_skip = self.relu(self.skip1(L1))
        L2_up = self.relu(self.deconv1(L2_out))
        L1_cat = torch.cat([L1_skip, L2_up], dim=1)
        L1_out = self.relu(self.upconv1(L1_cat))

        return L1_out


class UNet_down4(nn.Module):
    def __init__(self, nf=64):
        super(UNet_down4, self).__init__()
        # downsample
        self.conv2_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # skip
        self.skip1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.skip2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.skip3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsample
        self.deconv2 = nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.deconv1 = nn.ConvTranspose2d(nf, nf, 4, 2, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        L1 = x
        L2 = self.relu(self.conv2_2(self.relu(self.conv2_1(L1))))
        L3 = self.relu(self.conv3_2(self.relu(self.conv3_1(L2))))

        # L3
        L3_out = self.relu(self.skip3(L3))
        # L2
        L2_skip = self.relu(self.skip2(L2))
        L3_up = self.relu(self.deconv2(L3_out))
        L2_cat = torch.cat([L2_skip, L3_up], dim=1)
        L2_out = self.relu(self.upconv2(L2_cat))
        # L1
        L1_skip = self.relu(self.skip1(L1))
        L2_up = self.relu(self.deconv1(L2_out))
        L1_cat = torch.cat([L1_skip, L2_up], dim=1)
        L1_out = self.relu(self.upconv1(L1_cat))

        return L1_out


class Feature(nn.Module):
    def __init__(self, in_chl=3, nf=64, N_block=2, down=4):
        super(Feature, self).__init__()
        self.conv_first = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)

        block_f = functools.partial(block, nf=nf)
        self.fea_extract = make_layer(block_f, N_block)

        self.unet = eval('UNet_down%d(nf=nf)' % down)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv_first(x))
        x = self.fea_extract(x)
        out = self.unet(x)

        return out


if __name__ == '__main__':
    net = Feature()
    input = torch.randn(1, 3, 16, 16).float()
    output = net(input)
    print(output.size())
