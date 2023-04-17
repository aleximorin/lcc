import torch
import torchvision
from torch import functional as F

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_CHANNELS = (3, 64, 128, 256, 512, 1024)


class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):

        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Encoder(torch.nn.Module):

    def __init__(self, channels=None, kernel_size=3):

        super().__init__()

        if channels is None:
            channels = DEFAULT_CHANNELS
        self.channels = channels

        encoder_blocks = []
        for i, ch in enumerate(self.channels[:-1]):
            block = Block(self.channels[i], self.channels[i+1], kernel_size=kernel_size)
            encoder_blocks.append(block)

        self.encoder_blocks = torch.nn.ModuleList(encoder_blocks)
        self.pool = torch.nn.MaxPool2d(kernel_size - 1)

    def forward(self, x):
        features = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(torch.nn.Module):

    def __init__(self, channels=None, kernel_size=3):

        super().__init__()

        if channels is None:
            channels = DEFAULT_CHANNELS[1:][::-1]

        self.channels = channels

        upconvs = []
        decoder_blocks = []

        for i, ch in enumerate(self.channels[:-1]):
            up = torch.nn.ConvTranspose2d(self.channels[i], self.channels[i+1], kernel_size - 1, kernel_size - 1)
            dec = Block(self.channels[i], self.channels[i+1])

            upconvs.append(up)
            decoder_blocks.append(dec)

        self.upconvs = torch.nn.ModuleList(upconvs)
        self.decoder_blocks = torch.nn.ModuleList(decoder_blocks)

    def crop(self, x, encoder_features):
        _, _, H, W = x.shape
        out = torchvision.transforms.CenterCrop((H, W))(encoder_features)
        return out

    def forward(self, x, encoder_features):
        for i, ch in enumerate(self.channels[:-1]):
            x = self.upconvs[i](x)
            features = self.crop(x, encoder_features[i])
            x = torch.cat([x, features], dim=1)
            x = self.decoder_blocks[i](x)
        return x


class UNet(torch.nn.Module):

    def __init__(self, channels=None, kernel_size=3, num_class=2, sigmoid_max=1.0, sigmoid_min=-1.0):
        super().__init__()

        if channels is None:
            channels = DEFAULT_CHANNELS

        self.channels = channels
        self.encoder = Encoder(self.channels, kernel_size)
        self.decoder = Decoder(self.channels[1:][::-1], kernel_size)
        self.head = torch.nn.Conv2d(self.channels[1], num_class, 1)

        self.a = sigmoid_min
        self.b = sigmoid_max
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[-1], encoder_features[::-1][1:])
        out = self.head(out)
        out = self.sigmoid(out) * (self.b - self.a) - self.b

        return out


if __name__ == '__main__':

    # testing the different Modules of the UNET architecture
    block = Block(1, 64)
    x = torch.randn(1, 1, 572, 572)
    print(block.forward(x).shape)

    encoder = Encoder()
    x = torch.randn(1, 3, 572, 572)
    out = encoder.forward(x)
    for tmp in out:
        print(tmp.shape)

    decoder = Decoder()
    x = torch.randn(1, 1024, 28, 28)
    print(decoder.forward(x, out[::-1][1:]).shape)

    unet = UNet()
    x = torch.randn(1, 3, 572, 572)
    out = unet(x)
    print(out.shape)