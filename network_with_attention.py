import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.utils import _log_api_usage_once
# from ..ops.misc import MLP, Permute
# from ..ops.stochastic_depth import StochasticDepth
# from ..transforms._presets import ImageClassification, InterpolationMode
# from ..utils import _log_api_usage_once
# from ._api import register_model, Weights, WeightsEnum
# from ._meta import _IMAGENET_CATEGORIES
# from ._utils import _ovewrite_named_param, handle_legacy_interface
from ArcFace_loss import MultiLabelArcFaceLoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch

def _patch_merging_pad(x: torch.Tensor) -> torch.Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    return x


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x


class PatchMergingV2(nn.Module):
    """Patch Merging Layer for Swin Transformer V2.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)  # difference

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class FeatureAdjuster(nn.Module):  # 这个模块是将swin-transformer得到的特征图进行卷积得到统一的尺寸
    def __init__(self, target_channels=1024):
        super(FeatureAdjuster, self).__init__()
        self.conv_x1 = nn.Conv2d(256, target_channels, kernel_size=4, stride=4, padding=0)  # 从 64x64 缩小到 16x16
        self.conv_x2 = nn.Conv2d(512, target_channels, kernel_size=2, stride=2, padding=0)  # 从 32x32 缩小到 16x16
        self.conv_x3 = nn.Conv2d(1024, target_channels, kernel_size=1, stride=1, padding=0)  # 保持 16x16，只调整通道数

    def forward(self, features):
        # 调整维度顺序 b,h,w,c --> b,c,h,w
        x1, x2, x3 = (feature.permute(0, 3, 1, 2) for feature in features)
        
        # 调整特征
        x1 = self.conv_x1(x1)  # 调整 x1
        x2 = self.conv_x2(x2)  # 调整 x2
        x3 = self.conv_x3(x3)  # 调整 x3
         
        # 返回调整后的特征  b,c,h,w --> b,h,w,c
        x1, x2, x3 = (feature.permute(0, 2, 3, 1) for feature in (x1, x2, x3))
        return x1, x2, x3

class SelfAttention(nn.Module): 
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)  # 可选：添加 LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # 输入 x 的形状为 [Batch, Seq_Len, Channels]
        attn_output, _ = self.attention(x, x, x)  # 自注意力
        x = x + attn_output  # 残差连接
        x = self.norm(x)  # 可选：归一化
        mlp_output = self.mlp(x)  # MLP
        x = x + mlp_output  # 残差连接
        return x

class SelfAttentionProcessor(nn.Module):  # 利用自注意力机制对特征进行处理
    def __init__(self, target_channels=1024, num_heads=8):
        super(SelfAttentionProcessor, self).__init__()
        self.self_attention = SelfAttention(embed_dim=target_channels, num_heads=num_heads)

    def forward(self, features):
        # features 是一个包含 x1, x2, x3 的列表
        length = len(features)
        features = torch.cat(features, dim=0)  # 拼接所有特征为一个大张量 [Batch*Num_Features, C, H, W]
        batch, c, h, w = features.shape
        features = features.contiguous().view(batch, h * w, c)
        features = self.self_attention(features)
        features = features.contiguous().view(batch, c, h, w)
        # 拆分为单独特征
        outputs = torch.chunk(features, length, dim=0)

        return outputs

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输出形状为 [Batch, Channels, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),  # 恢复维度
            nn.Sigmoid(),  # 生成注意力权重
        )

    def forward(self, x):
        batch, channels, _, _ = x.shape
        y = self.avg_pool(x).view(batch, channels)  # [Batch, Channels]
        y = self.fc(y).view(batch, channels, 1, 1)  # 恢复形状为 [Batch, Channels, 1, 1]
        return x * y  # 权重与原始特征点乘

class ChannelAttentionProcessor(nn.Module):
    def __init__(self, channels):
        super(ChannelAttentionProcessor, self).__init__()
        self.channel_attention = ChannelAttention(in_channels=channels)

    def forward(self, features):
        # 堆叠输入特征列表，形状变为 [Batch*Num_Features, C, H, W]
        length = len(features)
        features = torch.cat(features, dim=0)

        # 应用通道注意力
        features = self.channel_attention(features)

        # 将处理后的张量拆分回列表
        outputs = torch.chunk(features, length, dim=0)
        return outputs  # 返回的是 [b, c, h, w] 的张量列表

class SwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: List[int] = [4, 4],
        embed_dim: int=128,
        depths: List[int] = [2, 2, 18, 2],
        num_heads: List[int] = [4, 8, 16, 32],
        window_size: List[int] = [7, 7],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 6,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        # 后面是修改添加的内容
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm2d(1024)
        self.bn3 = nn.BatchNorm2d(4096)
        self.adjust = FeatureAdjuster(target_channels=1024)
        self.self_attention = SelfAttentionProcessor(target_channels=1024, num_heads=8)
        self.channel_attention_process = ChannelAttentionProcessor(channels=1024)
        self.weights = nn.Parameter(torch.ones(4))  # Initialize weights for multi-scale fusion
        self.fusion_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.fc = nn.Linear(1024, 256)
        self.head = nn.Linear(256, num_classes)
        # ArcFaceLoss 初始化
        #self.arcface_loss = MultiLabelArcFaceLoss(num_classes=self.num_classes, embedding_size=embed_dim * 2 ** (len(depths) - 1))
        self.arcface_loss = MultiLabelArcFaceLoss(num_classes=self.num_classes, embedding_size=256)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        intermediate_features = []  # 保存中间层输出
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i%2 != 0 and i > 1:
                intermediate_features.append(x)  # 保存当前层输出
        #x = self.features(x)
        x1 = intermediate_features[0]  # Layer 0 output shape: torch.Size([1, 64, 64, 256])
        x2 = intermediate_features[1]  # Layer 1 output shape: torch.Size([1, 32, 32, 512])
        x3 = intermediate_features[2]   # Layer 2 output shape: torch.Size([1, 16, 16, 1024])
        x1,x2,x3 = self.adjust([x1,x2,x3])
        # 确保 x1, x2, x3, x 的形状是 b, c, h, w [1, 16, 16, 1024]形式

        combined = torch.cat([x1, x2, x3, x], dim=3)
        combined = combined.permute(0,3,1,2)
        combined = self.bn3(combined)
        x1, x2, x3, x = torch.chunk(combined, 4, dim=1)

        features = [x1, x2, x3, x] 
        features = self.self_attention(features) #b,c,h,w
        features = self.channel_attention_process(features) #b,c,h,w
        fused_features = sum(w * f for w, f in zip(self.weights, features))  # Weighted sum
        fused_features = self.bn2(fused_features)
        fused_features = self.fusion_conv(fused_features)  # Adjust channels[B 1024 16 16]
        fused_features = torch.mean(fused_features, dim=(2, 3))  # Shape: [B, 1024]
        fused_features = self.bn1(fused_features)

        feature = self.fc(fused_features)  # Shape: [B, 256]
        feature = self.relu(feature)
        x = self.head(feature) 

        return feature, x
    
    def compute_arcface_loss(self, feature, labels):
        """
        计算 ArcFace 损失。
        Args:
            feature (Tensor): 嵌入向量，形状为 [batch_size, embedding_size]。
            labels (Tensor): 标签，形状为 [batch_size]。
        """
        loss = self.arcface_loss(feature, labels)
        return loss

