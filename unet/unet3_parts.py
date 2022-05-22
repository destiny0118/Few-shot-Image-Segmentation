from torch import  nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    '''
        convolution->BN->ReLU
    '''
    def __init__(self,in_channels,out_channels,downSample=True,mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels=out_channels

        # 下采样
        if downSample:
            # 是否需要最后激活
            self.double_conv=nn.Sequential(
                nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 上采样
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.double_conv(x)

"""
    下采样，downsampling
    池化后进行两次卷积操作
"""
class DownSampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownSampling, self).__init__()
        self.pool_conv=nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.pool_conv(x)


"""
    上采样，upsampling
"""
class UpSampling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSampling, self).__init__()

        self.trans_conv=nn.Sequential(
            nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2,stride=2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.trans_conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x,input):
        # 对边缘补全
        diffY=input.size()[2]-x.size()[2]
        diffX=input.size()[3]-x.size()[3]

        x=F.pad(x,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
        return self.conv(x+input)
