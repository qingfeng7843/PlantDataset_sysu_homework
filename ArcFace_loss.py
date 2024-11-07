import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, s=10.0, m=0.50, easy_margin=False):
        """
        Args:
            embedding_size (int): 嵌入特征的维度。
            num_classes (int): 分类类别数。
            s (float): 缩放因子。
            m (float): 角度间隔。
            easy_margin (bool): 使用简化的边界条件。
        """
        super(MultiLabelArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.num_classes = num_classes
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(3.14159265 - m))  # 临界值 theta
        self.mm = torch.sin(torch.tensor(3.14159265 - m)) * m  # theta 的补偿值

        self.fc = nn.Parameter(torch.randn(embedding_size, num_classes))
        nn.init.xavier_uniform_(self.fc)

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): 模型的嵌入特征，形状为 (batch_size, embedding_size)。
            labels (Tensor): 多标签独热编码标签，形状为 (batch_size, num_classes)。
        Returns:
            Tensor: 计算得到的多标签 ArcFace 损失。
        """
        # L2 归一化嵌入特征和权重向量
        features = F.normalize(features, dim=1)
        weights = F.normalize(self.fc, dim=0)

        # 计算 cos(theta) 和 phi(theta)
        cosine = torch.matmul(features, weights)  # shape: (batch_size, num_classes)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # 应用 easy_margin 或 theta 边界条件
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 根据标签选择性应用 phi
        logits_arc = torch.where(labels == 1, phi, cosine)

        # 缩放 logits 并计算二元交叉熵损失
        logits_arc = logits_arc * self.s
        loss = F.binary_cross_entropy_with_logits(logits_arc, labels)

        return loss
