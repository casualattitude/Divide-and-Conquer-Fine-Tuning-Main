"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):##随机深度的dropout方法
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])##对应14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]##计算patch总数=14*14

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)##使用卷积将224*224*3变为14*14*768
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()##有传入norm_layer则使用，没有则不做操作

    def forward(self, x):##正向传播，将图片输入
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        ##输入图片尺寸不对就报错
        # flatten: [B, C, H, W] -> [B, C, HW]，[B,768，14*14]
        # transpose: [B, C, HW] -> [B, HW, C]，[B，14*14，768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)##对应patch embedding中的norm操作
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim=768
                 num_heads=8,##多头注意力机制的头数量
                 qkv_bias=False,##
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads##每个head的qkv维数就是整个维数除以head个数，对应word多头注意力图
        self.scale = qk_scale or head_dim ** -0.5##默认none则为注意力公式attention=q乘k转置除以根号dk乘v，一个norm处理
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)##用全连接层实现得到QKV
        self.attn_drop = nn.Dropout(attn_drop_ratio)##dropout
        self.proj = nn.Linear(dim, dim)##对应多头计算拼接后乘的Wo（让输出更好的融合起来）
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        # [batch_size, num_patches + 1（197）, total_embed_dim（768）]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]，相当于把qkv都均分为head数量份
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]，调整顺序，方便后面运算
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#3代表qkv3个参数，permute调整顺序。
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)，分别切片出q，k，v
        #print(q.shape)#[8,12,197,64]
        #k=k.permute(1,0,2,3)
        #print(k.shape)#[12,8,197,64]
        #print(v.shape)#[8, 12, 197, 64]
        #v=v.permute(1,0,2,3)
        #print(v.shape)#[12, 8, 197, 64]
        #k = k.reshape(1,self.num_heads,B*N,C // self.num_heads)##修改（1） k:[num_heads,B*N,embed_dim_per_head]
        #v = v.reshape(1,self.num_heads,B*N,C // self.num_heads)##修改（2） v:[num_heads,B*N,embed_dim_per_head]
        #print(k.shape)#[12,1576,64]
        #print(k.transpose(-2,-1).shape)#[12,64,1576]
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        #print(attn.shape)#[8,12,197,1576]
        #attn = (q @ k.transpose(-2, -1)) * self.scale##@只将k的后两个维度互换后对q和k进行矩阵乘，再除以根号dk    ~~~~~~~~~可修改位置
        attn = attn.softmax(dim=-1)#对每一行进行softmax操作
        attn = self.attn_drop(attn)##dropout操作
        #print(v.shape)#[12,1576,64]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        #print((attn @ v).shape)#[8,12,197,64]
        #print((attn @ v).transpose(1, 2).shape)#[8,197,12,64]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)##再与V进行矩阵乘，然后通过reshape将每一个头的结果拼接在一起
        x = self.proj(x)##拼接完有时候还需要通过Wo进行映射
        x = self.proj_drop(x)##dropout后输出
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        ##in_features：输入节点个数，hidden_features：第一个全连接层节点个数，是前一个的四倍，out_features与in_features相等，GELU激活函数
        super().__init__()
        out_features = out_features or in_features##默认和输入大小相同
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):##encoder block
    def __init__(self,
                 dim,##token的dim
                 num_heads,##head个数
                 mlp_ratio=4.,##第一个全连接层的节点个数，4代表乘4倍
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,##对应多头注意力后的dropout
                 attn_drop_ratio=0.,##对应q乘k的转置除以根号dk然后softmax后的dropout
                 drop_path_ratio=0.,##对应encoder block图中的dropout
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)##对应第一个layer norm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)##对应计算multi-head attention
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()##传入ratio>0则进行dropout，否则不操作
        self.norm2 = norm_layer(dim)##对应第二个layer norm
        mlp_hidden_dim = int(dim * mlp_ratio)##对应第一个全连接层的节点个数
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)##对应mlp block

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))##先通过norm1，再多头注意力，再dropout然后残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))##先通过norm2，再mlp，再dropout然后残差连接
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,##传入参数，depth：堆叠encoder block个数
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 ## representation_size对应最后的MLP head中的pre-logits全连接层节点个数，none则不构建，distilled是兼容搭建Deit模型
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes##分类的个数
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models，768
        self.num_tokens = 2 if distilled else 1##1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)##上面设为none所以用layernorm，partial方法传入默认参数eps
        act_layer = act_layer or nn.GELU##上面设为none所以默认用GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)##构建patch embedding
        num_patches = self.patch_embed.num_patches##得到patch个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))##初始化cls token第一个1是方便拼接用的，后两个代表cls的1*768
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None##默认不使用
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))##初始化position embedding，1，197，768
        self.pos_drop = nn.Dropout(p=drop_ratio)##对应加上position embedding后的dropout

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # 构建一个0到dropout ratio的等差序列，随着堆叠的encoder block深度变深而增大，默认为0， stochastic depth decay rule
        self.blocks = nn.Sequential(*[##对应堆叠的encoder block,*[]将list迭代出来，将列表打包为一个整体赋值给self.blocks
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)##对应通过encoder后的layer norm
        # Representation layer
        if representation_size and not distilled:##not distilled默认不用，有传入representation_size则会构建pre-logits层
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([##有构建则构建一个全连接层
                ("fc", nn.Linear(embed_dim, representation_size)),##embed_dim：输入的节点个数， representation_size：输出的节点个数
                ("act", nn.Tanh())##tanh作为激活函数
            ]))
        else:##传入none则不构建pre-logits层
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()##对应mlp head中的linear层
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]，对应patch embedding
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)##将cls复制B份，因为每张图都要一个cls
        if self.dist_token is None:##vit默认执行这行
            x = torch.cat((cls_token, x), dim=1)  # 将cls与x拼接 [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)##加上position embedding后dropout  B，197，768
        x = self.blocks(x)##通过堆叠的encoder block
        x = self.norm(x)##对应通过encoder block 后的layer norm
        if self.dist_token is None:##vit默认执行这行
            return self.pre_logits(x[:, 0])##第一个：代表取所有batch数据，第二个0代表索引0，也就是cls的位置，再通过pre-logits层
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:##默认为none所以不执行
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:##代表mlp head的linear得到最后的输出
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
