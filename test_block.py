from model import RepVGGBlock,Generator
from torch import nn
import torch

model = Generator(4).cuda()
model.eval()
# model.load_state_dict(torch.load('epochs_reparam_small/' + "netG_epoch_4_2.pth"))
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