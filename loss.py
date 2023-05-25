import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)#下载预训练模型，一般是在Imagenet数据集训练好的模型参数了
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()#loss_network是由VGG网络的前31个层（包括卷积层、池化层、ReLU层等）组成的序列模型，eval()进入评估模式，不计算该模型的反向传播、更新等工作
        for param in loss_network.parameters():
            param.requires_grad = False#将模型设置为评估模式并不会自动冻结所有参数，需要通过显式地将参数的requires_grad属性设置为False来冻结参数，以便它们在反向传播过程中不会被更新。
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss：GAN网络的损失
        adversarial_loss = torch.mean(1 - out_labels)#值越大，说明生成的效果不佳，被D判定为假
        # Perception Loss：输出图像LR和真实图像HR在VGG16特征图中的差异
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss：输出图像LR和真实图像HR本身在像素点上的差异值
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-6 * tv_loss
        # image_loss是内容损失，就是最普通的mseloss；adversarial_loss是gan网络的损失；perception_loss是VGG16的特征图损失；tv_loss是总变差损失

class TVLoss(nn.Module):#这段代码实现了一个总变差损失（Total Variation Loss），用于促进生成的图像平滑和连续，是一种无监督的正则化技术。
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):#x是out_images，也就是生成器生成的图像
        batch_size = x.size()[0]#batch_size
        h_x = x.size()[2]#height
        w_x = x.size()[3]#width
        count_h = self.tensor_size(x[:, :, 1:, :])# 表示从第 2 行开始到最后一行的所有像素点
        count_w = self.tensor_size(x[:, :, :, 1:])# 表示从第 2 列开始到最后一列的所有像素点
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()#计算高度方向上的像素差分，即沿着 height 维度方向上**相邻**两行像素值的差的平方和
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()#计算宽度方向上的像素差分，即沿着 width 维度方向上**相邻**两列像素值的差的平方和
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#最终，该函数的输出是一个标量值，表示总变差损失的大小。由于总变差损失没有目标图像，
#因此只需要在训练过程中将其作为一个正则化项添加到生成器的损失函数中即可。这样做可以有效提高生成图像的视觉质量和感知度。

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
