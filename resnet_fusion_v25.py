# --- resnet_fusion_v25.py (Model V2.5: ResNet-SAM + Tabular Fusion) ---
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
from cbam_model import load_resnet50_weights, SpatialAttention
import torch.nn.functional as F


# ----------------------------------------------------
# 1. 带 空间注意力 的 Bottleneck 块 (来自原 resnet_with_cbam.py)
# ----------------------------------------------------
class BottleneckSAM(Bottleneck):
    """继承 ResNet 的 Bottleneck 块，并在末尾添加 Spatial Attention 模块"""
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        # 调用父类 (Bottleneck) 的构造函数
        super(BottleneckSAM, self).__init__(inplanes, planes, stride, downsample,
                                            groups, base_width, dilation, norm_layer)
        # 空间注意力模块 (SAM)
        self.sam = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # ResNet Bottleneck 前三个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 1. 应用 SAM 模块 (在 Bottleneck 的末尾)
        out = self.sam(out)

        # 2. ResNet 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ----------------------------------------------------
# 2. ResNet50-SAM 骨干模型 (Model 2.0 基础)
# ----------------------------------------------------
class ResNet50_ImageBackbone(ResNet):
    """
    Model 2.0 图像特征提取器: ResNet-50 集成 Spatial Attention (仅图像输入)
    此版本将用于 V2.5 融合模型的骨干。
    """

    def __init__(self, num_classes=3, **kwargs):
        # 使用 BottleneckSAM 替换原始 Bottleneck
        super(ResNet50_ImageBackbone, self).__init__(BottleneckSAM, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

        # 权重加载 (与基线模型逻辑相同)
        num_ftrs = self.fc.in_features  # 2048
        custom_fc = nn.Linear(num_ftrs, num_classes)  # 用于加载预训练权重的占位符

        self.fc = nn.Linear(num_ftrs, 1000)
        load_resnet50_weights(self)  # 加载预训练权重
        self.fc = custom_fc  # 替换回 num_classes 的 FC，虽然融合模型会再替换它

        # 覆盖 ResNet 的 _forward_impl 方法
        self.forward_impl = super(ResNet50_ImageBackbone, self)._forward_impl

    def forward(self, img):
        x = img
        # 图像特征提取 -> AvgPool -> FC
        # 这里只返回 FC 前的特征 (如果是 ResNet 的默认 forward，它会返回最终分类结果)

        # 重新实现特征提取逻辑到 FC 之前 (确保只进行特征提取)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 不经过 self.fc
        return x


# ----------------------------------------------------
# 3. 最终融合模型 (Model V2.5)
# ----------------------------------------------------
class ResNetSAM_TabularFusion(nn.Module):
    """
    Model V2.5: 融合 2.0 骨干 (ResNet-SAM) 和 4.0 表格融合逻辑。
    """

    def __init__(self, num_classes, num_tabular=20):
        super(ResNetSAM_TabularFusion, self).__init__()

        # 1. 2.0 图像特征骨干 (ResNet50 + SAM)
        self.image_backbone = ResNet50_ImageBackbone(num_classes=num_classes)

        # 图像特征输出维度
        IMG_FTR_DIM = 2048  # ResNet-50 最终输出维度

        # 2. 4.0 表格特征处理网络 (来自 Model 4.0)
        self.fc_tabular = nn.Sequential(
            nn.Linear(num_tabular, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 512),  # 表格特征融合维度
            nn.ReLU(inplace=True)
        )

        # 3. 4.0 最终分类头
        FUSED_DIM = IMG_FTR_DIM + 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(FUSED_DIM, num_classes)
        )

    def forward(self, x_img, x_tabular):
        # 1. 提取图像特征 (Batch x 2048)
        image_features = self.image_backbone(x_img)

        # 2. 处理表格特征 (Batch x 512)
        tabular_out = self.fc_tabular(x_tabular)

        # 3. 特征融合 (拼接)
        fused_features = torch.cat((image_features, tabular_out), dim=1)

        # 4. 最终分类
        output = self.classifier(fused_features)

        return output