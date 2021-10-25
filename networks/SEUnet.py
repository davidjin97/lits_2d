import os
import torch
import torch.nn as nn

class SElayer(nn.Module):
    def __init__(self,channel,reduction=4):
        super(SElayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel)
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        y = torch.clamp(y,0,1)
        return x*y
 
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)
 
 
class SEUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEUnet, self).__init__()
 
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.se1 = SElayer(128)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2) 
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.se2 = SElayer(64)
        self.conv10 = nn.Conv2d(64, out_channels, 1)
 
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)# 128 128
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)# 64 64
        se1 = self.se1(p2)
        c3 = self.conv3(se1)
        p3 = self.pool3(c3)# 32 32
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)# 16 16
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1) #32
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1) #64
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1) #128
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1) #256
        c9 = self.conv9(merge9)
        se2 = self.se2(c9)
        c10 = self.conv10(se2)
        # out = nn.Sigmoid()(c10)
        return c10


if __name__ == "__main__":
    # model = SeUnet(1,2)
    # print(model)
    # input = torch.randn((8,1,256,256))
    # out = model(input)
    # print(out.shape)
    gpu_ids = "2, 3"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))

    model = SEUnet(1, 2).to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # input = torch.randn((2, 1, 64, 128, 160)).to(device) # BCDHW
    input = torch.randn(4, 1, 352, 352).to(device) # BCHW 
    output = model(input)

    print("input.shape: ", input.shape)
    print("output.shape: ", output.shape) # 2, 2, 32, 64, 64
    print(f"min: {output.min()}, max: {output.max()}")