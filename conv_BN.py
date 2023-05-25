from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def main():
    torch.random.manual_seed(0)

    f1 = torch.randn(1, 2, 3, 3)

    module = nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False),
        bn=nn.BatchNorm2d(num_features=2)
    ))
    # module = nn.Sequential(
    #     nn.Conv2d(in_channels=2,out_channels=2,kernel_size=3,padding=1,stride=1,bias=False),
    #     nn.BatchNorm2d(num_features=2)
    # )

    module.eval()

    with torch.no_grad():
        output1 = module(f1)
        print(output1)

    # fuse conv + bn
    kernel = module.conv.weight
    running_mean = module.bn.running_mean#bn.running_mean这是一个张量，它包含所有输入特征的均值。在每个训练批次中，该均值会被更新以反映当前批次中特征的均值。
    running_var = module.bn.running_var#bn.running_var：这也是一个张量，它包含所有输入特征的方差。类似于 bn.running_mean，该方差在每个训练批次中更新以反映当前批次中特征的方差。
    gamma = module.bn.weight#bn.weight：这是一个可学习的张量，它用于缩放 BN 层的输出。这个权重在训练过程中被优化，以便网络可以学习如何在不同的层之间适当地缩放特征。
    beta = module.bn.bias#bn.bias：这也是一个可学习的张量，它用于在 BN 层的输出中添加偏置。这个偏置在训练过程中被优化，以便网络可以学习如何在不同的层之间适当地添加偏置。
    eps = module.bn.eps#bn.eps：这是在计算 BN 层输出时用于数值稳定性的小数值。当计算方差时，我们需要除以样本数量，但如果样本数量非常小，那么方差会非常接近零，这可能会导致除零错误。为了避免这种情况，我们会在分母中添加一个很小的数值 eps。
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)  # [ch] -> [ch, 1, 1, 1]
    kernel = kernel * t
    bias = beta - running_mean * gamma / std
    fused_conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
    fused_conv.load_state_dict(OrderedDict(weight=kernel, bias=bias))

    with torch.no_grad():
        output2 = fused_conv(f1)
        print(output2)

    np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
    print("convert module has been tested, and the result looks good!")


if __name__ == '__main__':
    main()
