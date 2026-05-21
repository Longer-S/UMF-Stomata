
import torch
from torch import nn
from args_fusion import args
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def gradient_raw(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    #kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1);
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=1);
    #showTensor(gradMap);
    return gradMap;     


def gradient(x):
    """
    自适应梯度计算函数：支持 4D (B, C, H, W) 和 5D (B, N, C, H, W) 输入
    """
    # 记录原始维度数量
    original_dim = x.dim()
    
    if original_dim == 5:
        # 如果是 5D 堆栈: (B, N, C, H, W) -> (B*N, C, H, W)
        B, N, C, H, W = x.shape
        x_reshaped = x.view(B * N, C, H, W)
    elif original_dim == 4:
        # 如果是 4D 图像: (B, C, H, W)
        B, C, H, W = x.shape
        x_reshaped = x
    else:
        raise ValueError(f"Expected 4D or 5D input, but got {original_dim}D")

    device = x.device
    
    # 定义 Laplacian 卷积核 (用于提取边缘/梯度)
    kernel = torch.tensor([
        [0., 1., 0.],
        [1., -4., 1.],
        [0., 1., 0.]
    ], dtype=torch.float32, device=device)

    # 适配通道数 C
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # shape: [C, 1, 3, 3]

    # 使用 depthwise 卷积计算梯度
    grad = F.conv2d(x_reshaped, weight=kernel, padding=1, groups=C)

    # 恢复原始维度
    if original_dim == 5:
        grad = grad.view(B, N, C, H, W)
    # 如果是 4D，grad 已经是 (B, C, H, W)，无需操作
    
    return grad
     
def sumPatch(x, device, k):
    return F.avg_pool2d(x, kernel_size=2*k+1, stride=1, padding=k)






























