import torch 
import torch.nn as nn

class FocalLoss(nn.Module):
    # 初始化方法，定义损失函数的参数
    def __init__(self, alpha=0.2, gamma=2, logist=False, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 一个在0和1之间的标量因子，用于平衡损失
        self.gamma = gamma  # 焦点参数（始终为正），减小已正确分类示例的相对损失，更加关注难以分类的示例
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # 使用标准的交叉熵损失函数
        self.logist = logist  # 是否使用对数概率（未被使用）
        self.reduce = reduce  # 指定对输出应用的减少方式 - none/mean/sum
        # 'none': 不进行任何减少; 'mean': 输出的总和会被输出元素的数量除以; 'sum': 输出会被求和

    # 前向传播方法，计算损失
    def forward(self, inputs, targets):
        # 计算标准的交叉熵损失
        BCE_loss = self.cross_entropy_loss(inputs, targets)
        # 计算预测的概率
        pt = torch.exp(-BCE_loss)
        # 计算焦点损失
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 根据需要进行损失的减少
        if self.reduce:
            return torch.mean(F_loss)  # 返回损失的均值
        else:
            return F_loss  # 返回原始的损失值
