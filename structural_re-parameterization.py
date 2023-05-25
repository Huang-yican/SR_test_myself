import time
import torch.nn as nn
import numpy as np
import torch


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()

        if deploy:#不使用shortcut，直接使用一个2d卷积
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)#dilation是膨胀率

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=stride, padding=0, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):#rbr_reparam存在，说明是不使用shortcut，直接使用一个conv2d
            return self.nonlinearity(self.rbr_reparam(inputs))#对输入进行卷积+Relu即可

        #以下是对使用要做reparam的情况
        if self.rbr_identity is None:#对直接连接的处理
            id_out = 0#这会由广播机制补全的
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):#获取三个通道融合后等价的kernel和bias的方法
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):#1x1到3x3
        if kernel1x1 is None:
            return 0 #这会由广播机制补全的
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    #这段代码实现了将一个包含卷积层和批量归一化（BatchNorm）层的分支融合为一个等价的卷积层的功能
    def _fuse_bn_tensor(self, branch):#获取等价的kernel和bias的张量的实际实现过程
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):#如果这个分支为空，就返回0和0。如果分支是一个序列（Sequential），就从中获取卷积层的权重、批量归一化层的移动平均值、方差、伽马值和贝塔值以及批量归一化层的eps值；
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)#断点报错
            if not hasattr(self, 'id_tensor'):#BN层转为3*3的卷积
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1#中间位置是1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)#转换成张量类型，并将其发送到与branch.weight存储在同一设备上。这个张量会在之后的融合过程中使用
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        #下面是实现卷积+BN的融合
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):#重构为一个卷积
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
            #将神经网络模型中所有参数的梯度“detach”（截断）掉；下划线表示这个操作是原地进行的，也就是说，它会直接修改张量本身而不是创建一个新的副本
        #以下删除神经网络模型中的一些属性或变量，以便在需要时释放内存或清理对象
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.block1 = RepVGGBlock(in_channels=64, out_channels=64)
        self.block2 = RepVGGBlock(64, 64)
    def forward(self,x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        return out2


def main():
    f1 = torch.randn(1, 64, 64, 64)
    block = RepVGGBlock(64,64)
    block.eval()
    with torch.no_grad():
        output1 = block(f1)
        start_time = time.time()
        for _ in range(100):
            block(f1)
        print(f"consume time: {time.time() - start_time}")

        # re-parameterization
        block.switch_to_deploy()
        output2 = block(f1)
        start_time = time.time()
        for _ in range(100):
            block(f1)
        print(f"consume time: {time.time() - start_time}")

        np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
        print("convert module has been tested, and the result looks good!")


if __name__ == '__main__':
    main()
