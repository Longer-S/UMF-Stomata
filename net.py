import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxGuidedSpatialGating(nn.Module):
    def __init__(self, channels):
        super(MaxGuidedSpatialGating, self).__init__()
    
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True), 
            nn.Conv3d(channels, channels, kernel_size=1), 
            nn.Sigmoid() 
        )
    def forward(self, x):
        """
        x: (B, C, N, H, W)
        """
        max_val, _ = torch.max(x, dim=2, keepdim=True)

        max_val_expanded = max_val.expand_as(x)
        
        combined = torch.cat([x, max_val_expanded], dim=1)
        
        gate = self.gate_conv(combined)
        
        return x * gate

class Upsample3DBlock_3Dto3D(nn.Module):

    def __init__(self, in_channels_3d, skip_channels_3d, out_channels_3d):
        super().__init__()

        self.conv1_block = nn.Sequential(
            nn.ReflectionPad3d(1), 
            nn.Conv3d(
                in_channels_3d, 
                out_channels_3d,
                kernel_size=3, 
                padding=0, 
            )
        )
        self.bn1 = nn.BatchNorm3d(out_channels_3d)

        self.conv2_block = nn.Sequential(
            nn.ReflectionPad3d(1), 
            nn.Conv3d(
                out_channels_3d + skip_channels_3d, 
                out_channels_3d,
                kernel_size=3,
                padding=0, 
            )
        )
        self.bn2 = nn.BatchNorm3d(out_channels_3d)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_3d, skip_3d_pytorch):

        target_size = skip_3d_pytorch.shape[2:] 
        
        x_upsampled = F.interpolate(
            x_3d, 
            size=target_size,
            mode='trilinear', 
            align_corners=False
        )
        
        x_upsampled = self.relu(self.bn1(self.conv1_block(x_upsampled)))
        
        combined = torch.cat([x_upsampled, skip_3d_pytorch], dim=1)
        
        output = self.relu(self.bn2(self.conv2_block(combined)))
        return output

class AIFNet(nn.Module):
    def __init__(self, input_ch, output_ch, W=16, D=4, ret_bottleneck=False):
        super(AIFNet, self).__init__()
        self.D = D
        self.W = W
        self.ret_bottleneck = ret_bottleneck
        
        # 下采样卷积块: 
        self.conv_down = nn.ModuleList()
        # 第一层从 input_ch 到 W
        self.conv_down.append(self._conv_block(input_ch, W))
        # 后续层，每层 in/out 维度均为 W*(2**i)
        for i in range(1, D):
            channels = W * (2 ** i)
            self.conv_down.append(self._conv_block(channels, channels))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck: 输入通道为 W*(2**D)，输出相同
        channels_bottleneck = W * (2 ** D)
        # 保持 padding same
        self.bottleneck = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels_bottleneck, channels_bottleneck, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        # 定义 3D 空间门控
        self.spatial_gate = MaxGuidedSpatialGating(W * (2 ** D))
        self.up_blocks = nn.ModuleList()

        for i in range(D, 0, -1):
            # i = D, D-1, ..., 1
            # 输入通道来自上一个 3D 块 (或 3D bottleneck)
            in_ch = W * (2 ** i)
            # Skip 通道来自 Encoder
            skip_ch = W * (2 ** (i - 1))
            # 输出通道
            out_ch = W * (2 ** (i - 1))
            
            # 现在所有块都是 3D -> 3D
            block = Upsample3DBlock_3Dto3D(
                in_channels_3d=in_ch,
                skip_channels_3d=skip_ch,
                out_channels_3d=out_ch
            )
            self.up_blocks.append(block)
        
        self.conv_out_3d = nn.Sequential(
            nn.ReflectionPad3d(1), 
            nn.Conv3d(W, W, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.ReflectionPad3d(1), 
            nn.Conv3d(W, output_ch, kernel_size=3, stride=1, padding=0) 
        )


    def _conv_block(self, in_ch, out_ch):
        # 两个 conv + ReLU
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0),          
            nn.ReLU(),
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

    
    def forward(self, x):
        # x: B x FS x C x H x W
        B, FS, C, H, W = x.shape
        # 重塑为 B*FS, C, H, W
        h = x.view(B * FS, C, H, W)
        skip_feats = []  # 存储每层的 skip 特征 (B, ch, H_i, W_i)
        for i, conv in enumerate(self.conv_down):
            # 局部特征
            h = conv(h)  # shape: (B*FS, ch_i, H_i, W_i)
            # 池化
            pooled = self.pool(h)  # shape: (B*FS, ch_i, H_i/2, W_i/2)
            # 提取 skip: 先 reshape 到 (B, FS, ch_i, H_i, W_i)
            h_reshaped = h.view(B, FS, h.shape[1], h.shape[2], h.shape[3])
            # skip = torch.max(h_reshaped, dim=1)[0]  # (B, ch_i, H_i, W_i)
            # skip_feats.append(skip)
            skip_feats.append(h_reshaped)
            # 全局池化: (B, FS, ch_i, H_i/2, W_i/2)
            pooled_reshaped = pooled.view(B, FS, pooled.shape[1], pooled.shape[2], pooled.shape[3])
            global_max = torch.max(pooled_reshaped, dim=1)[0]  # (B, ch_i, H_i/2, W_i/2)
            # 扩展到每个 FS 元素: (B, FS, ch_i, H_i/2, W_i/2)
            global_expand = global_max.unsqueeze(1).expand(-1, FS, -1, -1, -1)
            # 重塑到 (B*FS, ch_i, H_i/2, W_i/2)
            global_expand = global_expand.reshape(B * FS, pooled.shape[1], pooled.shape[2], pooled.shape[3])
            # cat 本地池化和全局信息
            h = torch.cat([pooled, global_expand], dim=1)
            # 下一层 conv_down 会接收正确的通道数
        # Bottleneck
        h = self.bottleneck(h)  # (B*FS, ch_b, H_b, W_b)

        # reshape 到 3D
        h_reshaped = h.view(B, FS, h.shape[1], h.shape[2], h.shape[3]) # (B, FS, C_b, H_b, W_b)
        
        
        # 1. 将 bottleneck 3D 特征转换为 PyTorch 格式 (B, C, N, H, W)
        h_3d = h_reshaped.permute(0, 2, 1, 3, 4) # (B, C_b, FS, H_b, W_b)

        h_3d = self.spatial_gate(h_3d)

        # 2. 逐级 3D 上采样
        for i, block in enumerate(self.up_blocks):
            
            # 获取 3D skip (B, N, C, H, W)
            skip_user_format = skip_feats.pop(-1)
            
            # 转换为 PyTorch 格式 (B, C, N, H, W)
            skip_pytorch = skip_user_format.permute(0, 2, 1, 3, 4)
            
            # 3D -> 3D 上采样
            h_3d = block(h_3d, skip_pytorch) 
        
        # 循环结束后, h_3d 是 (B, W, N, H, W)
        
        cost = self.conv_out_3d(h_3d) # (B, output_ch, N, H, W)
        cost=cost.permute(0, 2, 1, 3, 4)
        pred_mask=F.softmax(cost, dim=1)
        output = torch.sum(x * pred_mask, dim=1)
        return output,cost,pred_mask

if __name__ == '__main__':

    net = AIFNet(input_ch=3, output_ch=1, W=16, D=4).cuda()
    
    # 输入: (Batch=2, Stack=5, Channel=3, H=128, W=128)
    x = torch.randn(2, 5, 3, 128, 128).cuda()
    
    # 接收两个返回值
    fused_image, mask = net(x)
    
    print('Fused Image shape:', fused_image.shape) # 期望: (2, 3, 128, 128)
    print('Mask shape:       ', mask.shape)        # 期望: (2, 5, 1, 128, 128)
