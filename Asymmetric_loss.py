import torch
import torch.nn as nn

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma_pos=2, gamma_neg=4, reduce='sum'):
        super(AsymmetricFocalLoss, self).__init__()
        self.alpha = alpha  # 平衡因子
        self.gamma_pos = gamma_pos  # 正类的焦点参数
        self.gamma_neg = gamma_neg  # 负类的焦点参数
        self.reduce = reduce  # 损失的减少方式

    def forward(self, inputs, targets):
        # inputs: Sigmoid 后的概率值
        # targets: 真实标签 (0 或 1)

        # 计算交叉熵损失
        BCE_loss = - (targets * torch.log(inputs + 1e-8) + (1 - targets) * torch.log(1 - inputs + 1e-8))
        
        # 直接使用输入的预测概率作为 pt
        pt = inputs * targets + (1 - inputs) * (1 - targets)

        # 根据正负类选择不同的 gamma 值
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)

        # 计算 Asymmetric Focal Loss
        AFL_loss = self.alpha * (1 - pt) ** gamma * BCE_loss

        # 返回损失
        if self.reduce == 'mean':
            return AFL_loss.mean()
        elif self.reduce == 'sum':
            return AFL_loss.sum()
        else:
            return AFL_loss



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()