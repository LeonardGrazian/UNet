
import torch
from torch import nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class EncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        full = self.double_conv(x)
        pooled = self.pool(full)
        pooled = self.dropout(pooled)
        return full, pooled


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, add_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.dropout = nn.Dropout(0.3)
        self.double_conv = DoubleConvBlock(
            out_channels + add_channels,
            out_channels
        )

    def forward(self, x, s):
        x = self.conv_transpose(x)
        x = torch.cat([x, s], dim=1)
        x = self.dropout(x)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        encoder_channels,
        bottleneck_channels,
        decoder_channels
    ):
        super().__init__()

        assert len(encoder_channels) == 4
        assert len(decoder_channels) == 4

        # define hyperparameters
        self.enc1_ch = encoder_channels[0]
        self.enc2_ch = encoder_channels[1]
        self.enc3_ch = encoder_channels[2]
        self.enc4_ch = encoder_channels[3]

        self.btnk_ch = bottleneck_channels

        self.dec1_ch = decoder_channels[0]
        self.dec2_ch = decoder_channels[1]
        self.dec3_ch = decoder_channels[2]
        self.dec4_ch = decoder_channels[3]

        # define model
        self.enc1 = EncodeBlock(3, self.enc1_ch)
        self.enc2 = EncodeBlock(self.enc1_ch, self.enc2_ch)
        self.enc3 = EncodeBlock(self.enc2_ch, self.enc3_ch)
        self.enc4 = EncodeBlock(self.enc3_ch, self.enc4_ch)

        self.btnk = DoubleConvBlock(self.enc4_ch, self.btnk_ch)

        self.dec1 = DecodeBlock(self.btnk_ch, self.enc4_ch, self.dec1_ch)
        self.dec2 = DecodeBlock(self.dec1_ch, self.enc3_ch, self.dec2_ch)
        self.dec3 = DecodeBlock(self.dec2_ch, self.enc2_ch, self.dec3_ch)
        self.dec4 = DecodeBlock(self.dec3_ch, self.enc1_ch, self.dec4_ch)

        self.final = nn.Conv2d(self.dec4_ch, 3, 1, padding='same')

    def forward(self, x):
        # encoder
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)

        # bottleneck
        btnk = self.btnk(p4)

        # decoder
        dec1 = self.dec1(btnk, f4)
        dec2 = self.dec2(dec1, f3)
        dec3 = self.dec3(dec2, f2)
        dec4 = self.dec4(dec3, f1)

        Y = self.final(dec4)
        return Y
