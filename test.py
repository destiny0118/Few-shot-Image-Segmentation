from  unet import UNet


if __name__=="__main":
    fp = open("test.txt", "wb")
    # print(input, file=fp)
    # print(target, file=fp)
    print("*****", fp)
    print("********")
    fp.close()
    # net = UNet(n_channels=3, n_classes=2, bilinear=False)