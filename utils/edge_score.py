from torchvision import transforms
from PIL import Image
from torch import Tensor
import torch
from matplotlib import pyplot as plt
import numpy as np

'''
    对图像分割边缘划分的准确率
'''
def edge_loss(mask_predit:Tensor,mask_true:Tensor,device,weight=50,true_edge=True):
    '''
        将分割图中的实例部分为1，其余为0
    '''
    # print(mask_true.size(),mask_predit.size())
    # tensor=tensor.to(device)
    cmp=mask_true>0
    cmp=cmp+0
    cmp=cmp.clone().type(torch.int)

    if len(list(cmp.shape))==2:
        # 左右填充数据
        l_r_p = torch.zeros((cmp.shape[0], 1), dtype=torch.int).cuda()

        # 上下填充数据
        u_d_p = torch.zeros((1, cmp.shape[1]), dtype=torch.int).cuda()

        '''
           获取填充后向各个方向移位的tensor
        '''
        left_1 = torch.cat((cmp, l_r_p), 1)[:, 1:].cuda()
        right_1 = torch.cat((l_r_p, cmp), 1)[:, :-1].cuda()
        up_1 = torch.cat((cmp, u_d_p), 0)[1:, :].cuda()
        down_1 = torch.cat((u_d_p, cmp), 0)[:-1, :].cuda()

    elif len(list(cmp.shape)) == 3:
        l_r_p = torch.zeros((1, cmp.shape[1], 1), dtype=torch.int).cuda()
        u_d_p = torch.zeros((1, 1, cmp.shape[2]), dtype=torch.int).cuda()

        left_1 = torch.cat((cmp, l_r_p), 2)[:, :, 1:].cuda()
        right_1 = torch.cat((l_r_p, cmp), 2)[:, :, :-1].cuda()
        up_1 = torch.cat((cmp, u_d_p), 1)[:, 1:, :].cuda()
        down_1 = torch.cat((u_d_p, cmp), 1)[:, :-1, :].cuda()



    '''
        获取边缘
    '''
    a = torch.bitwise_xor(cmp, left_1).cuda()
    b = torch.bitwise_xor(cmp, right_1).cuda()
    c = torch.bitwise_xor(cmp, up_1).cuda()
    d = torch.bitwise_xor(cmp, down_1).cuda()

    # ans=a+b+c+d
    ans = torch.bitwise_or(a, b).cuda()
    ans = torch.bitwise_or(ans, c).cuda()
    ans = torch.bitwise_or(ans, d).cuda()

    edge=ans.cuda()
    if(true_edge):
        edge=torch.bitwise_and(cmp,edge)

    # print(edge.sum())
    '''
        给边缘加权
    '''
    # TP = edge * mask_predit
    # print(TP.sum(), edge.sum(),TP.sum()/edge.sum())
    edge=edge*weight
    TP=edge*mask_predit
    # print(TP.sum(),edge.sum(),TP.sum()/edge.sum())

    return 1-TP.sum()/edge.sum()

    '''
        转换为PIL格式图片
    '''
    # to_image=transforms.ToPILImage()
    # img=to_image(edge)
    # img.save("edge.gif")
    # plt.imshow(img)
    # plt.show()




    # if(save):


def image_trans(img_path):
    img=Image.open(img_path)
    # PIL格式转化为ndarray格式
    img_narray=np.asarray(img)
    if(img_narray.ndim>2):
        img_narray=img_narray.transpose((2,0,1))

    # img_narray=img_narray/255
    img_tensor=torch.as_tensor(img_narray)
    print(type(img_tensor))

    # plt.imshow(img)
    # plt.show()
    return img_tensor


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tesnsor=image_trans("1_mask.gif")

    img_tesnsor=img_tesnsor.to(device=device)

    print(img_tesnsor.size())
    edge_loss(img_tesnsor,img_tesnsor,device,weight=50)
