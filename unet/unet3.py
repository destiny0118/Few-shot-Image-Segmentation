import torch
from .unet3_parts import *
class UNet3(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet3,self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes

        self.inc=DoubleConv(n_channels,64,mid_channels=32)

        self.down1=DownSampling(64,128)
        self.down2=DownSampling(128,256)
        self.down3=DownSampling(256,512)
        self.down4=DownSampling(512,1024)


        self.up1=UpSampling(1024,512)
        self.up2=UpSampling(512,256)
        self.up3=UpSampling(256,128)
        self.up4=UpSampling(128,64)


        self.outc=OutConv(64,n_classes)

    def forward(self,x):
        input=self.inc(x)

        x=self.down1(input)
        x=self.down2(x)
        x=self.down3(x)
        x=self.down4(x)

        y=self.up1(x)
        y=self.up2(y)
        y=self.up3(y)
        y=self.up4(y)

        # res=y4
        logits=self.outc(y,input)
        return logits



if __name__=="__main__":
    x=torch.rand(1,3,959,640)
    net=ResUnet(3,2)
    y=net(x)
    print(y.shape)





