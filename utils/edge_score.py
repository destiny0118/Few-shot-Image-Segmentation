from torchvision import transforms
from PIL import Image
from torch import Tensor
import torch
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

'''
    对图像分割边缘划分的准确率
    edge_weight: 边缘权值
    outline_weight: 轮廓权重
'''


def get_edge(mask_true: Tensor, save_mask=False):
    '''
        将分割图中的实例部分为1，其余为0
    '''

    cmp = mask_true > 0
    cmp = cmp + 0
    cmp = cmp.clone().type(torch.int)

    if len(list(cmp.shape)) == 2:
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

    '''边缘和轮廓'''
    outline_edge = ans.cuda()
    edge = torch.bitwise_and(cmp, outline_edge)
    outline = torch.bitwise_xor(outline_edge, edge)

    # 转换为PIL格式图片
    if save_mask:
        to_image = transforms.ToPILImage()
        img = to_image(edge)
        img.save("edge.gif")
        plt.imshow(img)
        plt.show()

    return edge, outline


def edge_coeff(input: Tensor, target: Tensor, epison=1e-6):
    assert input.size() == target.size(),f"input size is {input.size()} target size is {target.size()}"
    # 遍历channel，计算每个channel的损失
    if (input.dim() == 2):
        cross_sum = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(target)
        # print("cross_sum:",cross_sum,"sets_sum:",sets_sum)
        return cross_sum / sets_sum
    else:
        score = 0
        for channel in range(input.shape[0]):
            score += edge_coeff(input[channel,...], target[channel,...], epison)
        return score / input.shape[0]


def multiclass_edge_coeff(input: Tensor, target: Tensor, edgeWeight: float = 1, outlineWeight: float = 1, epison=1e-6):
    # 得到每个样本的边缘加权表示
    # 计算一个批次中的平均损失
    score = 0
    for i in range(input.shape[0]):
        # 得到边缘one_hot码并加权
        edge, outline = get_edge(target[i, ...])
        # edge=torch.tensor(edge.copy())
        # print("edge:",edge)
        # print("outline:",outline)
        # edge = F.one_hot(edge, 2)
        # outline = F.one_hot(outline, 2)
        # tmp = torch.ones_like(outline)
        # outline = torch.bitwise_xor(outline, tmp)
        edge=torch.unsqueeze(edge,dim=0)*edgeWeight
        outline=torch.unsqueeze(outline,dim=0)*outlineWeight
        res = torch.cat((outline,edge),0)
        # print(res.size(),input[i].size(),res)

        score += edge_coeff(input[i, ...], res.float())
    return score / input.shape[0]


def edge_loss(input: Tensor, target: Tensor, edge_weight,outline_weight,multiclass: bool = False):
    #    input: [batch, channels, h, w]
    #    target：[batch, h, w]
    assert input.dim()==4,f"Expected 4 dim, but got input.size:{input.size()}"
    assert target.dim() == 3, f"Expected 3 dim, but got target.size:{target.size()}"
    fn = multiclass_edge_coeff if multiclass else edge_coeff
    return 1 - fn(input, target,edge_weight,outline_weight)


def image_trans(img_path):
    img = Image.open(img_path)
    # PIL格式转化为ndarray格式
    img_narray = np.asarray(img)
    if (img_narray.ndim > 2):
        img_narray = img_narray.transpose((2, 0, 1))

    # img_narray=img_narray/255
    img_tensor = torch.as_tensor(img_narray)
    print(type(img_tensor))

    # plt.imshow(img)
    # plt.show()
    return img_tensor


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # img_tesnsor = image_trans("1_mask.gif")
    #
    # img_tesnsor = img_tesnsor.to(device=device)
    #
    # print(img_tesnsor.size())

    true_mask = torch.tensor([[[0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 1, 1, 1, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0]]], dtype=torch.int)

    pred_mask = torch.tensor([
        [[[0, 0, 1, 0, 0],
          [0, 1, 0, 1, 0],
          [1, 0, 1, 0, 1],
          [0, 1, 0, 1, 0],
          [0, 0, 1, 0, 0]],

         [[0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0]]]

    ]
        , dtype=torch.float)
    true_mask = true_mask.to(device=device)
    pred_mask = pred_mask.to(device=device)
    loss=edge_loss(pred_mask, true_mask,  edge_weight=10, outline_weight=1,multiclass=True)
    print(loss)
