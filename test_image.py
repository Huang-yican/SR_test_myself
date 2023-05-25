import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator,RepVGGBlock

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name',default="zebra.png", type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')#选用模型G的参数
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False#True就是GPU，False就是CPU
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name  #模型参数

model = Generator(UPSCALE_FACTOR).eval()#推理预测不计算梯度了
if TEST_MODE:#GPU
    model.cuda()
    model.load_state_dict(torch.load('epochs_reparam/' + MODEL_NAME))
else:#CPU
    model.load_state_dict(torch.load('epochs_reparam/' + MODEL_NAME, map_location=lambda storage, loc: storage))#lambda storage, loc: storage表示将所有存储器加载到默认设备（即CPU）上

image = Image.open("D:\\desktop\\test\\" + IMAGE_NAME)
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)#volatile=True参数指示变量在推理时不需要计算梯度。unsqueeze(0)函数将封装的张量对象增加了一个维度，以便于后续的操作
if TEST_MODE:
    image = image.cuda()

start = time.time()
out = model(image)
elapsed = (time.time() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('D:\\desktop\\test\\srgan_result\\' + str(UPSCALE_FACTOR) + '_voc2012_res16_' + IMAGE_NAME)
