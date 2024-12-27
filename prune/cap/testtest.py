import timm
import torch

# 加载 ResNet-50D 模型
model = timm.create_model('vit_base_patch16_224', pretrained=False)

# 打印模型结构
print(model)