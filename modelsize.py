# import torch
# from thop import clever_format, profile_origin
# from ECB6_ESA6 import Generator

# # 定义模型
# model = Generator(4)

# # 输入示例数据
# input_size = (3, 1024, 1024)  # 输入图像的尺寸
# input_data = torch.randn((1,) + input_size)

# # 计算模型的乘加操作数量
# flops, params = profile_origin(model, inputs=(input_data,))

# print(f"乘加操作数量: {flops / 1e9} Gflops")
# print(f"模型参数数量: {params / 1e6} Mparams")

import torch
import torchstat
from ECB6_ESA6 import Generator

# 定义模型
model = Generator(4)
model.load_state_dict(torch.load('epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_128/netG_best_4.pth'))

# 输入示例数据
input_size = (3, 256, 256)  # 输入图像的尺寸

# 打印模型的乘加操作数量
flops, params = torchstat.stat(model, input_size=input_size)
print(f"乘加操作数量: {flops}")
print(f"模型参数数量: {params}")
