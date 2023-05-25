from ECB6_ESA6 import Generator
from torch import nn
import torch
import argparse
import time
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import os
import cv2
import numpy as np
import math

model = Generator(4).cuda(1)
model.eval()
model.load_state_dict(torch.load('epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_512_finetune/netG_best_4.pth'))
# if isinstance(model.block2, RepVGGBlock):
#     print("Model contains MyBlock!")

# def is_my_block(module):
#     if hasattr(module, "__dict__") and "name" in module.__dict__ and module.__dict__["name"] == "re_param":
#         return True
#     else:
#         return False

# for name, module in model.named_modules():
#     if is_my_block(module):
#         print(name)
#         module.switch_to_deploy()


print('# generator parameters:', sum(param.numel() for param in model.parameters()))#G的参数数量

# 定义存储PSNR值的列表
psnr_list = []
folder_path = "./set14"
start = time.time()
img_number = 0
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path)
    

    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)#volatile=True参数指示变量在推理时不需要计算梯度。unsqueeze(0)函数将封装的张量对象增加了一个维度，以便于后续的操作
    image = image.cuda(1)
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    
    # image_truth = Image.open("./test/SRF_4/target/"+filename)

    # # 将 PIL.Image.Image 对象转换为 numpy 数组，并提取 Y 通道数据
    # orig_y = np.asarray(out_img.convert('YCbCr'))[..., 0]
    # comp_y = np.asarray(image_truth.convert('YCbCr'))[..., 0]

    # # 计算 MSE 和 PSNR
    # mse = np.mean((orig_y - comp_y) ** 2)
    # psnr = 10 * np.log10((255 ** 2) / mse)

#region
    # image_truth = Variable(ToTensor()(image_truth), volatile=True)
    # image_truth = image_truth.cuda()
    # img_mse = ((out - image_truth) ** 2).data.mean()
    # psnr = 10 * math.log10((image_truth.max()**2) / (img_mse))

    # psnr = cv2.PSNR(out_img,image_truth)#使用这个指令的话需要先转换为numpy
#endregion

    # psnr_list.append(psnr)
    # out_img.save('./test_results/' + str(4) + '_ECB6_ESA6_UNet_L1_DIV2K_finetune/' + filename)
    img_number += 1


#region
# 计算平均PSNR值
# avg_psnr = sum(psnr_list) / len(psnr_list)
#endregion

elapsed = (time.time() - start)
print('cost' + str(elapsed/img_number) + 's')
# average = sum(psnr_list)/len(psnr_list)
# print(average)