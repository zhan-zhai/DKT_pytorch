import torch


#  将源域数据和目标域数据转化为核矩阵
#  kernel_num: 表示的是多核的数量
#  fix_sigma: 表示是否使用固定的标准差
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total扩展
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 计算高斯核中的|x-y|^2
    L2_distance = ((total0-total1)**2).sum(2)

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|^2/bandwith)

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # 将多个核合并在一起


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
#     print(source.size(),target.size())
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Source<->Source[0-batch_size,0-batch_size]
    YY = kernels[batch_size:, batch_size:]  # Target<->Target
    XY = kernels[:batch_size, batch_size:]  # Source<->Target
    YX = kernels[batch_size:, :batch_size]  # Target<->Source
    loss = torch.mean(XX + YY - XY -YX)
    return loss

import torch
import torch.nn.functional as F

def rbf_mmd(x, y, gamma):
    # 计算样本 x 和 y 的 RBF MMD

    # 计算 x 和 y 的核矩阵
    Kxx = torch.exp(-gamma * torch.cdist(x, x)**2)
    Kxy = torch.exp(-gamma * torch.cdist(x, y)**2)
    Kyy = torch.exp(-gamma * torch.cdist(y, y)**2)

    # 计算 MMD 的两个部分
    m = x.shape[0]
    n = y.shape[0]
    mmd_x = (1.0 / (m * (m - 1))) * torch.sum(Kxx) - (2.0 / (m * n)) * torch.sum(Kxy) + (1.0 / (n * (n - 1))) * torch.sum(Kyy)
    mmd_y = (1.0 / (n * (n - 1))) * torch.sum(Kyy) - (2.0 / (m * n)) * torch.sum(Kxy) + (1.0 / (m * (m - 1))) * torch.sum(Kxx)

    # 取 MMD 的平均值作为最终的距离度量
    mmd = 0.5 * (mmd_x + mmd_y)

    return mmd