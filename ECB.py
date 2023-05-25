import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeqConv3x3(nn.Module):  # 各个分支结构
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):  # 分支类型、输入通道、输出通道、通道变化（文中一般是2）
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)  # scale就是边缘提取器这一分支的可学习参数
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt=False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x) + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

#region 以下是ESA模块的实现
def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)#还原为原来通道数
        m = self.sigmoid(c4)
        # print(m.shape)
        return x * m
#endregion



class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))#基于给定的缩放因子（scale_factor）计算出超分辨率神经网络中需要的上采样块（upsample block）的数量。
#所以放大倍数不同，如放大4倍和放大8倍相比，网络模型在block8的层数会相差一层，所以可以训练多轮数据后，根据实际需要调用相应的网络模型参数文件
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ECB(64, 64,2)
        self.block3 = ECB(64, 64,2)
        self.esa1 = ESA(64)        
        self.block4 = ECB(64, 64,2)
        self.block5 = ECB(64, 64,2)
        self.esa2 = ESA(64)
        self.block6 = ECB(64, 64,2)
        self.block7 = ECB(64, 64,2)
        self.esa3 = ESA(64)
        self.block8 = ECB(64, 64,2)
        self.block9 = ECB(64, 64,2)
        self.esa4 = ESA(64)
        self.block10 = ECB(64, 64,2)
        self.block11 = ECB(64, 64,2)
        self.esa5 = ESA(64)
        self.block12 = ECB(64, 64,2)
        self.block13 = ECB(64, 64,2)
        self.esa6 = ESA(64)
        self.block14 = ECB(64, 64,2)
        self.block15 = ECB(64, 64,2)
        self.esa7 = ESA(64)
        self.block16 = ECB(64, 64,2)
        self.block17 = ECB(64, 64,2)#这里修改为16个repVGG block
        self.esa8 = ESA(64)
        self.block18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block19 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]#这里调用UpsampleBLock类时，参数是64，2，其中2就是对应上一条注释，每次上采样扩大2倍
        block19.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block19 = nn.Sequential(*block19)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        esa1 = self.esa1(block3)
        block4 = self.block4(esa1)
        block5 = self.block5(block4)
        esa2 = self.esa2(block5)
        block6 = self.block6(esa2)
        block7 = self.block7(block6)
        esa3 = self.esa3(block7)        
        block8 = self.block8(esa3)
        block9 = self.block9(block8)
        esa4 = self.esa4(block9)
        block10 = self.block10(esa4)
        block11 = self.block11(block10)
        esa5 = self.esa5(block11)
        block12 = self.block12(esa5)
        block13 = self.block13(block12)
        esa6 = self.esa6(block13)
        block14 = self.block14(esa6)
        block15 = self.block15(block14)        
        esa7 = self.esa7(block15)
        block16 = self.block16(esa7)
        block17 = self.block17(block16)
        esa8 = self.esa8(block17)
        block18 = self.block18(esa8)

        #
        block19 = self.block19(block1 + block18)#注意这里也有一个残差

        return (torch.tanh(block19) + 1) / 2

"""Leaky ReLU中的ai是固定的；
    RReLU中的aji是一个在一个给定的范围内随机抽取的值，这个值在测试环节就会固定下来。"""


#判别器有点过于复杂化，降低
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),#512，512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),#512，1024
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)#1024，1
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))#最后输出的是经过sigmoid的操作


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)#亚像素卷积，像素上采样：尺寸变大，通道变少，例如当up_scale=2时，输入张量的每个维度的尺寸将乘以2，而通道数将变为原来的1/4
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
if __name__ == '__main__':
    x = torch.randn(1,3,88,88)
    net = Generator(4)
    y = net(x)
    # print(x)
    # print(y)
    print(x.shape)
    print(y.shape)