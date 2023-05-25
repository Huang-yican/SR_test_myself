from os import listdir
from os.path import join
import torchvision.transforms.functional as F
import random

from PIL import Image,ImageDraw#Python Imaging Library 的 Image 模块，用于图像的读取、处理、保存等。
from torch.utils.data.dataset import Dataset#PyTorch 提供的数据集类，用于加载数据集并进行数据处理。
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
#torchvision.transforms：PyTorch 提供的数据转换类，用于对图像进行常用的数据增强和预处理操作，如裁剪、缩放、旋转等。
##Compose：将多个数据预处理操作合并成一个处理流程。
##RandomCrop：对图像进行随机裁剪。
##ToTensor：将 PIL.Image 对象或 numpy.ndarray 转换为 PyTorch 的 tensor 对象。
##ToPILImage：将 tensor 对象转换为 PIL.Image 对象。
##CenterCrop：对图像进行中心裁剪。
##Resize：对图像进行缩放

from torch.utils.data import DataLoader

def is_image_file(filename):#判断是否图片
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):#用于计算经过裁剪后的图像尺寸，使其可以被上采样因子整除
    return crop_size - (crop_size % upscale_factor)
#crop_size 是指在进行训练和验证时对输入图像进行的裁剪操作的尺寸；upscale_factor 是指需要将输入图像上采样的倍数

def train_hr_transform(crop_size):#不需要上采样
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])
def train_hr_crop(crop_size):
    return Compose([
        RandomCrop(crop_size)
    ])
def train_hr_tensor():
    return Compose([
        ToTensor()
    ])

def train_lr_transform(crop_size, upscale_factor):#训练数据LR预处理，需要上采样
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),#值得注意的是，这里的crop_size已经经过calculate_valid_crop_size（）函数操作了
        ToTensor()
    ])

def flu_transform(inputs):
    # 高斯模糊处理
    blur_tensor = F.gaussian_blur(inputs, kernel_size=5, sigma=1)
    return blur_tensor

#它将图像转换为PIL图像，然后调整大小，居中裁剪并转换为张量。
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

#定义一个随机生成直线的函数
def draw_lines(img):
   # 获取图像大小
    width, height = img.size
    # 创建用于绘制的对象
    draw = ImageDraw.Draw(img)
    # 随机生成直线的起始点和结束点
    x1 = random.randint(0, width - 1)
    y1 = random.randint(0, height - 1)
    x2 = random.randint(0, width - 1)
    y2 = random.randint(0, height - 1)
    # 随机生成直线宽度
    # width = random.randint(10, 15)
    width = 10
    # 画直线
    draw.line((x1, y1, x2, y2), fill='black', width=width)
    # 保存修改后的图像
    return img


class TrainDatasetFromFolder(Dataset):#在创建对象时需要传入一个 dataset_dir、截取图像大小、超分倍数，该目录下应包含所有要用于验证的图像文件。
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.train_hr_crop = train_hr_crop(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    # def __getitem__(self, index):
    #     hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
    #     #引入高斯卷积
    #     blu_image = flu_transform(hr_image)
    #     lr_image = self.lr_transform(blu_image)
    #     #lr_image = self.lr_transform(hr_image)
    #     return lr_image, hr_image

    def __getitem__(self, index):
        hr_image_PIL = Image.open(self.image_filenames[index])
        hr_image = self.hr_transform(hr_image_PIL)#这是为了生成对应的lr
        #引入高斯卷积
        blu_image = flu_transform(hr_image)
        lr_image = self.lr_transform(blu_image)
        hr_image_crop = self.train_hr_crop(hr_image_PIL)
        hr_image_line = draw_lines(hr_image_crop)
        # hr_image_line = train_hr_tensor(hr_image_line)

        return lr_image, ToTensor()(hr_image_line)

    #返回数据集的大小，它被 DataLoader 对象用来确定有多少个 batch
    def __len__(self):
        return len(self.image_filenames)


#创建验证数据集
class ValDatasetFromFolder(Dataset):#在创建对象时需要传入一个 dataset_dir，该目录下应包含所有要用于验证的图像文件。
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    # def __getitem__(self, index):
    #     hr_image = Image.open(self.image_filenames[index])
    #     w, h = hr_image.size
    #     crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)#这里指定大小为该图w、h的短边，且满足可以整除upscale_factor
    #     lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)#lr要除以放大倍数
    #     hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)#hr则是直接resize为指定的大小
    #     hr_image = CenterCrop(crop_size)(hr_image)#在原图中间上截取指定大小的图片，也就是HR
    #     lr_image = lr_scale(hr_image)
    #     hr_restore_img = hr_scale(lr_image)#经过下采样和上采样后的低分辨率图像，
    #     return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)#这里指定大小为该图w、h的短边，且满足可以整除upscale_factor
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)#lr要除以放大倍数
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)#hr则是直接resize为指定的大小
        hr_image = CenterCrop(crop_size)(hr_image)#在原图中间上截取指定大小的图片，也就是HR
        lr_image = lr_scale(hr_image)
        # #多加了一项line
        # hr_image = draw_lines(hr_image)
        hr_restore_img = hr_scale(lr_image)#经过下采样和上采样后的低分辨率图像，
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    ##返回四个值：分别是图像的名称 image_name，原始的低分辨率图像 lr_image，经过插值放大之后的低分辨率图像 hr_restore_img，以及对应的高分辨率图像 hr_image

    def __len__(self):
        return len(self.lr_filenames)

if __name__ == '__main__':
    train_dataset = TrainDatasetFromFolder("data_mini/train_HR",400,4)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=2, shuffle=False)
    i = 1
    for data, target in train_dataloader:
        to_pil = ToPILImage()
        data_img = to_pil(data[0])
        print(type(data_img))
        data_img.save("data_mini/test/test_img.jpg")
        target_img = to_pil(target[0])
        target_img.save("data_mini/test/test_target{}.jpg".format(i))
        i += 1
        print("yes!")
        print(data.shape)
        print(target.shape)