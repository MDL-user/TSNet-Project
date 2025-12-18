# --- cbam_model.py (Model 2.0 空间注意力架构基础) ---
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet
import torch.nn.functional as F


# ----------------------------------------------------
# 辅助函数：加载预训练权重
# ----------------------------------------------------
def load_resnet50_weights(model: ResNet):
    """加载 ResNet-50 预训练权重并处理 fc 层"""
    num_ftrs = model.fc.in_features
    resnet50_url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    state_dict = load_state_dict_from_url(resnet50_url, progress=True)

    original_fc = model.fc
    model.fc = nn.Linear(num_ftrs, 1000)
    model.load_state_dict(state_dict, strict=False)
    model.fc = original_fc


# ----------------------------------------------------
# 空间注意力模块 (Spatial Attention Module, SAM)
# ----------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入通道为 2 (Avg Pool + Max Pool)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 通道维度的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 2. 拼接 (Concat)
        x_out = torch.cat([avg_out, max_out], dim=1)

        # 3. 卷积和 Sigmoid 激活得到空间注意力图
        attention_map = self.sigmoid(self.conv1(x_out))

        # 4. 空间加权
        return x * attention_map