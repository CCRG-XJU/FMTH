import torch
from torch import nn
from timm.layers import SelectAdaptivePool2d
from einops import rearrange
import torch.nn.init as init
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum
import math
import numpy as np
import einops
from torch.autograd import Function
from torch.nn.init import calculate_gain
from einops.layers.torch import Rearrange
from models_fusion import CustLayerNorm


# 全局缓存字典，用于存储预计算的DCT滤波器
dct_filters_cache = {}

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 768:
            self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 9, stride=1, padding=16, groups=dim, dilation=4)
        if dim == 384:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        if dim == 192:
            self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
        else:
            self.conv0 = nn.Conv2d(dim, dim, 1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        # self.conv3 = nn.Conv2d(dim, 1, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
    
class LSKAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # x = rearrange(x, 'b h w c -> b c h w')
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        # x = rearrange(x, 'b c h w -> b h w c')
        return x 

class FrequencyDistributionLearner(nn.Module):
    def __init__(self, channels, height, width, M, level):
        super(FrequencyDistributionLearner, self).__init__()
        self.M = M
        self.h = height
        self.w = width
        self.level = level
        # 初始化学习过滤器并转移到合适的设备
        initial_learnable_filters = torch.zeros(M, 1, height, width, 1).cuda()  # 初始化形状为 [M, 1, H, W, 1]
        self.learnable_filters = nn.Parameter(initial_learnable_filters)  # 将整个张量作为一个参数
        
        # 初始化固定过滤器
        fixed_filters = torch.zeros(M, 1, height, width, 1).cuda()
        mask_ = torch.zeros(height, width).cuda()
        diagonals4 = [min(height, width) * (i + 1) // M for i in range(M)]
        for i, diagonal in enumerate(diagonals4): #diagonals4 diagonals6
            mask = torch.from_numpy(np.triu(np.ones((height, width)), k=height-2 * diagonal)).float()
            mask = torch.fliplr(mask)
            fixed_filters[i, 0, :, :, 0] = mask
        self.fixed_filters = fixed_filters
        
        self.weight = MultiSpectralDCTLayer(height, width, channels)
        
    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x.view(B, C, H, W, 1) #diagonals6
        x = self.weight(x)
        x = x.view(B, C, H, W, 1)  # 整合reshape和unsqueeze操作
        # print(x.shape)
        
        # 应用tanh激活函数到学习过滤器并与固定过滤器结合
        learnable_filters_tanh = torch.tanh(self.learnable_filters)
        if self.level == 0:
            filtered = x + x * (self.fixed_filters[0] + learnable_filters_tanh[0])  # 扩展x以应用所有过滤器
        if self.level == 1:
            filtered = x + x * (self.fixed_filters[1] + learnable_filters_tanh[1])  # 扩展x以应用所有过滤器
        if self.level == 2:
            filtered = x + x * (self.fixed_filters[2] + learnable_filters_tanh[2])  # 扩展x以应用所有过滤器
        if self.level == 3:
            filtered = x + x * (self.fixed_filters[3] + learnable_filters_tanh[3])  # 扩展x以应用所有过滤器

        return filtered.squeeze(-1)  # 移除最后一个维度

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate DCT filters in PyTorch, with caching to avoid redundant computations.
    """
    def __init__(self, height, width, dim):
        super(MultiSpectralDCTLayer, self).__init__()
        cache_key = (height, width, dim)
        if cache_key not in dct_filters_cache:
            dct_filters_cache[cache_key] = self.get_dct_filter(height, width, dim).cuda()
        self.register_buffer('dct_weight', dct_filters_cache[cache_key])

    def forward(self, x):
        b, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # Change to PyTorch's default shape (N, C, H, W)
        assert x.dim() == 4, 'x must be 4 dimensions, but got ' + str(x.dim())
        x = x * self.dct_weight
        return x.permute(0, 2, 3, 1)  # Change back to the original shape

    def build_filter(self, pos, freq, POS):
        """
        Vectorized build filter function for DCT.
        """
        factor = np.pi / POS
        return torch.cos(factor * (pos + 0.5) * freq) / np.sqrt(POS)

    def get_dct_filter(self, tile_size_x, tile_size_y, dim):
        dct_filter = torch.zeros(dim, tile_size_x, tile_size_y, device='cuda')

        # 预计算C_u和C_v因子
        C_u = torch.tensor([1 / np.sqrt(2) if u_x == 0 else 1 for u_x in range(tile_size_x)], device='cuda')
        C_v = torch.tensor([1 / np.sqrt(2) if v_y == 0 else 1 for v_y in range(tile_size_y)], device='cuda')

        # 为tile_size_x和tile_size_y生成位置索引
        u_x = torch.arange(tile_size_x, device='cuda').view(tile_size_x, 1)
        v_y = torch.arange(tile_size_y, device='cuda').view(1, tile_size_y)
        
        # 使用向量化操作预计算DCT基函数
        u_filter = self.build_filter(u_x, u_x.t(), tile_size_x)  # 使用转置来计算外积
        v_filter = self.build_filter(v_y, v_y.t(), tile_size_y)

        # 计算二维DCT滤波器的外积
        for i in range(dim):
            dct_filter[i] = torch.matmul(u_filter * C_u.view(-1, 1), v_filter * C_v)
        
        # 归一化因子
        norm_factor = 2 / torch.sqrt(torch.tensor(tile_size_x * tile_size_y, dtype=torch.float32, device='cuda'))
        dct_filter *= norm_factor

        return dct_filter


if __name__ == '__main__':
    pass
