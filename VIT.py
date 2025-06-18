import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class DynamicDepthwiseConv(nn.Module):
    """动态卷积核生成+深度卷积"""
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 动态生成卷积核参数
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, kernel_size**2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 生成动态卷积核权重 [b, k*k, 1, 1]
        kernel_weights = self.attention(x).view(b, 1, self.kernel_size**2, 1, 1)
        
        # 展开输入特征图进行卷积操作
        unfolded_x = F.unfold(x, kernel_size=self.kernel_size, 
                            padding=self.padding, stride=self.stride)
        unfolded_x = unfolded_x.view(b, c, self.kernel_size**2, h//self.stride, w//self.stride)
        
        # 应用动态权重 [b, c, k*k, h', w'] * [b, 1, k*k, 1, 1]
        weighted_x = unfolded_x * kernel_weights
        out = torch.sum(weighted_x, dim=2)
        return out
    
class EnhancedDepthwiseSeparableFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., kernel_size=3, expansion_ratio=4):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        
        # Expanded point-wise convolution
        self.pw_expand = nn.Sequential(
            nn.Conv2d(dim, dim*expansion_ratio, 1),
            nn.GELU(),
            nn.BatchNorm2d(dim*expansion_ratio)
        )
        
        # Depthwise Convolution with dynamic kernel
        self.dw_conv = nn.Sequential(
            DynamicDepthwiseConv(dim*expansion_ratio, kernel_size=kernel_size),
            nn.GELU(),
            nn.BatchNorm2d(dim*expansion_ratio),
            ChannelAttention(dim*expansion_ratio)  # 通道注意力
        )
        
        # Projection
        self.pw_project = nn.Sequential(
            nn.Conv2d(dim*expansion_ratio, dim, 1),
            nn.BatchNorm2d(dim)
        )
        
        # Skip connection with stochastic depth
        self.drop_path = nn.Dropout2d(p=dropout) if dropout > 0. else nn.Identity()
        
        # Spatial Attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: (batch, seq_len, dim)"""
        b, n, d = x.shape
        h = w = int(math.sqrt(n))
        residual = x
        
        # Reshape to 4D
        x = x.view(b, h, w, d).permute(0, 3, 1, 2)  # [b, d, h, w]
        
        # Expansion
        x = self.pw_expand(x)
        
        # Depthwise Conv with attention
        x = self.dw_conv(x)
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        # Projection
        x = self.pw_project(x)
        
        # Reshape back
        x = x.permute(0, 2, 3, 1).reshape(b, n, d)
        
        # Stochastic depth and residual
        return residual + self.drop_path(x)

# class DepthwiseSeparableFFN(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0., kernel_size=3):
#         super().__init__()
#         self.dim = dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
        
#         # 第一阶段：逐点升维 (1x1卷积实现全连接)
#         self.pw_conv1 = nn.Conv2d(
#             in_channels=dim, 
#             out_channels=hidden_dim, 
#             kernel_size=1
#         )
        
#         # 深度可分离卷积 (分组数=输入通道数)
#         self.dw_conv = nn.Conv2d(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim,
#             kernel_size=kernel_size,
#             padding=kernel_size//2,  # 保持特征图尺寸不变
#             groups=hidden_dim,       # 深度卷积关键参数
#             bias=False
#         )
        
#         # 第二阶段：逐点降维 (1x1卷积实现全连接)
#         self.pw_conv2 = nn.Conv2d(
#             in_channels=hidden_dim,
#             out_channels=dim,
#             kernel_size=1
#         )
        
#         # 激活函数与正则化
#         self.gelu = nn.GELU()
#         self.dropout = nn.Dropout(dropout)
        
#         # 动态获取特征图尺寸的辅助参数
#         self._init_shape_params()

#     def _init_shape_params(self):
#         """用于处理动态特征图尺寸的中间缓存"""
#         self.cached_h = None
#         self.cached_w = None

#     def _get_spatial_size(self, seq_len):
#         """自动推导特征图尺寸 (假设输入为正方形)"""
#         h = w = int(seq_len**0.5)
#         if h * w != seq_len:
#             raise ValueError(f"序列长度{seq_len}不是完美平方数，无法推导空间尺寸")
#         return h, w

#     def forward(self, x):
#         """输入形状: (batch, seq_len, dim)"""
#         b, n, d = x.shape
        
#         # 自动获取或验证空间尺寸
#         if self.cached_h is None or self.cached_w is None:
#             h, w = self._get_spatial_size(n)
#             self.cached_h, self.cached_w = h, w
#         else:
#             h, w = self.cached_h, self.cached_w
        
#         # 转换为卷积需要的4D形状: (batch, channels, height, width)
#         x = x.view(b, h, w, d).permute(0, 3, 1, 2)  # [b, d, h, w]
        
#         # 逐点升维
#         x = self.pw_conv1(x)          # [b, hidden_dim, h, w]
#         x = self.gelu(x)
#         x = self.dropout(x)
        
#         # 深度卷积
#         x = self.dw_conv(x)           # [b, hidden_dim, h, w]
#         x = self.gelu(x)
#         x = self.dropout(x)
        
#         # 逐点降维
#         x = self.pw_conv2(x)          # [b, dim, h, w]
#         x = self.dropout(x)
        
#         # 恢复序列形状
#         x = x.permute(0, 2, 3, 1)     # [b, h, w, dim]
#         x = x.reshape(b, n, d)        # [b, seq_len, dim]
        
#         return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, EnhancedDepthwiseSeparableFFN(
                    dim, 
                    hidden_dim=mlp_dim,
                    dropout=dropout,
                    kernel_size=3,
                    expansion_ratio=4
                ))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, word_size,num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., transformer = None, t2t_layers = ((7, 4), (3, 2), (3, 2))):
        super().__init__()
        layers = []
        layer_dim = channels
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)

            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim = layer_dim, heads = 1, depth = 1, dim_head = layer_dim, mlp_dim = layer_dim, dropout = dropout) if not is_last else nn.Identity(),
            ])
        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        word_height, word_width = pair(word_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        #self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), 
        #    nn.Linear(patch_dim, dim),
        #)
        self.to_patch_embedding_ = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), 
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2, dim))

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ConvTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'cls' else x[:, 0]

        x = self.to_latent(x)
        return 1