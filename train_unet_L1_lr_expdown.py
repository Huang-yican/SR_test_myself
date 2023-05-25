import argparse# Python 自带的命令行参数解析模块
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils# 包含了用于图像操作和可视化的工具类和函数
from torch.autograd import Variable# PyTorch的Variable类，提供了用于自动求导的接口
from torch.utils.data import DataLoader
from tqdm import tqdm #第三方库，可以用于显示进度条。

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss_unet_L1 import GeneratorLoss
from ECB6_ESA6 import Generator, Discriminator#值得注意的是，这里的model就必须从ECB.py的G和D
from torch.optim.lr_scheduler import StepLR

#定义了一个ArgumentParser对象，用于解析命令行参数，可以根据参数指定的类型和值来生成帮助文档，还可以将命令行参数转换为 Python 对象
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=600, type=int, help='train epoch number')#先跑个10轮


if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size#88
    UPSCALE_FACTOR = opt.upscale_factor#upscale_factor是指放大倍数，与model中的缩放因子scale_factor是一个东西
    NUM_EPOCHS = opt.num_epochs#epoch数
    device = torch.device("cuda",0)
    batch_size = 84

    #预处理、读取数据（训练train，验证val）
    train_set = TrainDatasetFromFolder('data_DIV2K/train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data_DIV2K/val_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)#batch_size改小，原来是64，12用了2.6g，可以考虑20
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)#train中数据是打乱输入的，val中没有
    
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))#G的参数数量
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))#D的参数数量
    
    generator_criterion = GeneratorLoss()#定义生成器的损失
    
    if torch.cuda.is_available():#如果GPU可用，见G\D模型，以及G的损失函数移到GPU上
        netG.to(device)
        netD.to(device)
        generator_criterion.to(device)
    
    # optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
    # schedulerG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.995)
    # optimizerD = optim.Adam(netD.parameters(), lr=0.0001)
    # schedulerD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.995)
       
    num = int(len(train_set) / batch_size)
    # print(num)
    
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001)
    schedulerG = StepLR(optimizerG, step_size= 100*num, gamma=1 / 3)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0001)
    schedulerD = StepLR(optimizerD, step_size= 100*num, gamma=1 / 3)

    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    # results字典中存储了6个key：d_loss，g_loss，d_score，g_score，psnr，ssim
    
    for epoch in range(1, NUM_EPOCHS + 1):#遍历num_epoch次
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()#开始训练
        netD.train()
        for data, target in train_bar:#train_bar是一个可以实现进度条效果的迭代器，每次迭代会产生一个batch的训练数据，一个一个batch传递数据
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
    
            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target)# Variable 是将 Tensor 包装成变量，这样可以跟踪计算图的构建，并自动计算梯度。但是在新的pytorch中，可以不用了，tensor直接就可以包含了 Variable的所有功能
            if torch.cuda.is_available():
                real_img = real_img.to(device)
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.to(device)
            fake_img = netG(z)
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            #这里定义了d_loss
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()#更新D权值
            schedulerD.step()
            
    
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            #netD返回的是一个包含每个样本判别结果的张量，.mean()操作将该张量中所有的结果进行平均，得到一个标量值作为这个批次中所有样本的平均判别结果，然后再用该值更新判别器网络的参数
            
            
            optimizerG.step()#更新G权值
            schedulerG.step()

            # loss for current batch before optimization
            # item() 方法将 tensor 中的值作为 Python 数字返回，这里乘以 batch size 是为了按照 batch size 进行累加，方便后续计算平均值。
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))#输出目前该batch的结果
        
        current_lr = optimizerG.param_groups[0]['lr']
        print(f"Epoch: {epoch}, current lr: {current_lr}")

        netG.eval()#创建文件夹SRF_
        out_path = 'training_results/SRF_reparam_ECB6ESA6_DIV2K_unet_L1_imgsize_128_lr_expdown' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        #验证模型
        with torch.no_grad():
            val_bar = tqdm(val_loader)#一个一个batch的验证数据
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr#LR图片
                hr = val_hr#HR图片
                if torch.cuda.is_available():
                    lr = lr.to(device)
                    hr = hr.to(device)
                sr = netG(lr)#根据之前训练的G网络，利用lr生成对应的sr，每跑完一个epoch就生成一次
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()#计算HR和SR的ssim
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))#显示这次epoch运行的结果
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])# .squeeze(0)是一个PyTorch张量方法，用于删除张量中大小为1的维度。在这里，.squeeze(0)用于删除张量中大小为1的批次维度。因此，当张量的大小为[1, C, H, W]时，.squeeze(0)将其大小更改为[C, H, W]

            if epoch % 50 == 0:
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)#将这个大张量拆分成多个包含15个子张量的张量列表
                val_save_bar = tqdm(val_images, desc='[saving training results]')#显示结果，从左到右分别是lr（放大为了一起显示对比），HR，SR（实验结果）
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)#调用torchvision.utils.make_grid函数来将一批图像合并成一个网格，并将其保存到指定的输出路径中。其中，nrow参数指定每一行显示的图像数量，padding参数指定图像之间的填充量
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1
    
        # save model parameters，每50个epoch都保存D和G的参数到epochs文件夹中
        if epoch % 50 == 0:
            torch.save(netG.state_dict(), 'epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_128_lr_expdown/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
            torch.save(netD.state_dict(), 'epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_128_lr_expdown/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        if valing_results['psnr'] >= max(results['psnr']):
            torch.save(netG.state_dict(), 'epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_128_lr_expdown/netG_best_%d.pth' % (UPSCALE_FACTOR))
            torch.save(netD.state_dict(), 'epochs_reparam_ESB6_ESA6G_epoch600_DIV2K_unet_L1_imgsize_128_lr_expdown/netD_best_%d.pth' % (UPSCALE_FACTOR))
            print("==================save the best result===================save the best result===================save the best result===================")
    
        if epoch % 20 == 0 and epoch != 0:#每20个epoch后将训练结果写入statistics/csv文件中
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results_reparam_ECB6ESA6_epoch600_DIV2K_unet_L1_imgsize_128_lr_expdown.csv', index_label='Epoch')












