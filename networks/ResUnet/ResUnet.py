import os
import sys
import torch
import torch.nn as nn
from networks.ResUnet.modules import ResidualConv, Upsample

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output

if __name__ == "__main__":
    gpu_ids = "2, 3"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))

    model = ResUnet(1, 2).to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # input = torch.randn((2, 1, 64, 128, 160)).to(device) # BCDHW
    input = torch.randn(4, 1, 352, 352).to(device) # BCHW 
    output = model(input)

    print("input.shape: ", input.shape)
    print("output.shape: ", output.shape) # 2, 2, 32, 64, 64
    print(f"min: {output.min()}, max: {output.max()}")