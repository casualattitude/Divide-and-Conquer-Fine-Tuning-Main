import torch

# 加载.pth文件
path_to_pth_file = "research/transimagenet_experiment/transimagenet_experiment/vit_base_patch16_224_sparsity=0.00_best.pth"
model_state_dict = torch.load(path_to_pth_file, map_location=torch.device('cpu'))

# 打印模型的键值对
print("Keys in the state dict:")
for key in model_state_dict.keys():
    print(key)
