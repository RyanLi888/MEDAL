"""
MADE (Masked Autoencoder for Distribution Estimation) 核心实现
===============================================================

真正的 MADE 模型，用于密度估计。
通过掩码机制确保自回归性质，能够有效建模高维数据的条件分布。

主要特性：
1. 掩码线性层：确保自回归性质
2. 支持高斯输出（输出 mu 和 log_var）
3. 基于负对数似然 (NLL) 的密度估计

参考实现: https://github.com/e-hulten/made

"""

from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU


class MaskedLinear(nn.Linear):
    """
    带掩码的线性变换层
    
    通过掩码机制限制权重矩阵的连接，确保自回归性质。
    公式: y = x.dot(mask*W.T) + b
    """

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        """
        初始化掩码线性层
        
        参数:
            n_in: 输入样本的维度
            n_out: 输出样本的维度
            bias: 是否包含偏置项，默认True
        """
        super().__init__(n_in, n_out, bias)
        # 使用 register_buffer 确保 mask 随模型移动到正确设备
        self.register_buffer('mask', torch.ones(n_out, n_in))

    def initialise_mask(self, mask: Tensor):
        """
        初始化掩码
        
        参数:
            mask: 掩码张量
        """
        self.mask.copy_(mask)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播：应用掩码线性变换
        
        参数:
            x: 输入张量
            
        返回:
            掩码后的线性变换结果
        """
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """
    MADE (Masked Autoencoder for Distribution Estimation) 主模型
    
    该模型通过掩码机制确保自回归性质，能够有效建模高维数据的条件分布。
    支持高斯输出，适用于密度估计任务。
    
    核心原理：
    - 每个输出维度只依赖于输入中索引小于它的维度
    - 通过掩码矩阵实现这种依赖关系的约束
    - 高斯模式下输出 mu 和 log_var，用于计算概率密度
    """
    
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = True,
        random_order: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        初始化MADE模型
    
        参数:
            n_in: 输入维度
            hidden_dims: 隐藏层维度列表
            gaussian: 是否使用高斯MADE（输出mu和log_var），默认True
            random_order: 是否使用随机输入排序，默认False
            seed: numpy随机种子，默认None
        """
        super().__init__()
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 模型参数
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in  # 高斯输出维度翻倍
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []

        # 构建层维度列表
        dim_list = [self.n_in, *hidden_dims, self.n_out]

        # 创建网络层
        layers = []
        for i in range(len(dim_list) - 2):
            layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]))
            layers.append(ReLU())

        # 最后一层：隐藏层到输出层
        layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))

        # 创建顺序模型
        self.model = nn.Sequential(*layers)

        # 为掩码激活创建掩码
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 (B, n_in)
            
        返回:
            如果 gaussian=True: (B, 2*n_in) - 前半部分是 mu，后半部分是 log_var
            如果 gaussian=False: (B, n_in) - sigmoid 后的概率
        """
        if self.gaussian:
            # 高斯输出：返回原始的 mu 和 log_var
            return self.model(x)
        else:
            # 伯努利输出：通过 sigmoid 将概率压缩到 (0,1) 区间
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """
        为隐藏层创建掩码
        
        该方法实现了MADE的核心机制：通过掩码确保自回归性质。
        每个隐藏单元只能连接到输入层中索引小于等于其连接数的单元。
        """
        # 定义常量以提高可读性
        L = len(self.hidden_dims)  # 隐藏层数量
        D = self.n_in              # 输入维度

        # 决定是否使用随机或自然输入排序
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # 为隐藏层设置连接数 m
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        # 为输出层添加 m，输出顺序与输入顺序相同
        self.masks[L + 1] = self.masks[0]

        # 为 输入->隐藏1->...->隐藏L 创建掩码矩阵
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            
            # 初始化掩码矩阵
            M = torch.zeros(len(m_next), len(m))
            for j in range(len(m_next)):
                # 使用广播比较 m_next[j] 与 m 中的每个元素
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int))
            
            # 添加到掩码矩阵列表
            self.mask_matrix.append(M)

        # 如果输出是高斯分布，将输出单元数量翻倍（mu, log_var）
        # 成对相同的掩码
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # 用掩码初始化 MaskedLinear 层
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))
    
    def compute_log_prob(self, x: Tensor) -> Tensor:
        """
        计算输入样本的对数概率密度
        
        参数:
            x: 输入张量 (B, n_in)
            
        返回:
            log_prob: (B,) 每个样本的对数概率密度
        """
        if not self.gaussian:
            raise ValueError("compute_log_prob only works with gaussian=True")
        
        output = self.forward(x)  # (B, 2*n_in)
        mu, log_var = torch.chunk(output, 2, dim=1)  # 各 (B, n_in)
        
        # 限制 log_var 范围，防止数值不稳定
        log_var = torch.clamp(log_var, min=-10, max=10)
        var = torch.exp(log_var)
        
        # 高斯对数似然: log p(x) = -0.5 * sum( (x-mu)^2/var + log_var + log(2*pi) )
        log_prob = -0.5 * torch.sum(
            torch.pow(x - mu, 2) / var + log_var + np.log(2 * np.pi),
            dim=1
        )
        
        return log_prob
    
    def compute_nll_loss(self, x: Tensor) -> Tensor:
        """
        计算负对数似然损失 (Negative Log-Likelihood)
        
        参数:
            x: 输入张量 (B, n_in)
            
        返回:
            nll: scalar - 批次平均 NLL 损失
        """
        log_prob = self.compute_log_prob(x)
        return -log_prob.mean()
