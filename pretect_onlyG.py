import argparse
import time
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from ECB6_ESA6 import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--device', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_dir',default='miniLED/led_seg', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_512_finetune/netG_best_4.pth', type=str, help='generator model epoch name')
parser.add_argument('--save_dir',default='test_results/4_ECB6_ESA6_UNet_L1_DIV2K',type=str,help='the test results save here')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
IMAGE_DIR = opt.image_dir
MODEL_NAME = opt.model_name
SAVE_DIR = opt.save_dir


model = Generator(UPSCALE_FACTOR).cuda()
model.load_state_dict(torch.load(MODEL_NAME))
num_image = len(os.listdir(IMAGE_DIR))
start_time = time.time()

for filename in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(image_path)

    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)  # volatile=True参数指示变量在推理时不需要计算梯度。unsqueeze(0)函数将封装的张量对象增加了一个维度，以便于后续的操作
    image = image.cuda()
    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(SAVE_DIR+'/'+filename)

end_time = time.time()
print("the sum of the time: {}, and every image average cost {}s".format(end_time-start_time,(end_time-start_time)/num_image))