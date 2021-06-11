import os
import json
import argparse
import numpy as np
import math
from random import shuffle
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from carn.dataset import TestDataset
from model.carn import Net
from carn.sample import save_image
import pytorch_ssim
from matplotlib import pyplot
import torch_dct as dct


PSNR_LR = []
PSNR_SR = []

pyplot.rcParams['font.sans-serif'] = ['KaiTi']


def attack_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="carn")
    parser.add_argument("--ckpt_path", type=str, default="../checkpoint/carn.pth")
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--sample_dir", type=str, default="../dataset/test")  # 结果保存地址
    parser.add_argument("--test_data_dir", type=str, default="../dataset/urban100")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--alpha", type=int, default=32/255)  # 像素值最大变化
    parser.add_argument("--T", type=int, default=50)   # 攻击迭代次数
    parser.add_argument("--type", type=str, default='white')
    # 攻击类型   白盒:white   白盒局部:part   黑盒:black
    parser.add_argument("--part_size", type=int, default=10)  # 局部攻击大小
    parser.add_argument("--picture_quality", type=str, default='L2')  # 损失函数 ，L2或ssim
    return parser.parse_args()


def black_attack(net, device, dataset, cfg, alpha, T):
    loss_fun = nn.MSELoss()
    part_size = cfg.part_size

    for step, (hr, lr, name) in enumerate(dataset):
        # print(lr.size())
        # print(lr.numel())
        print('=====================图片:{}====================='.format(name))
        lr = lr.unsqueeze(0).to(device)
        sr0 = net(lr, cfg.scale).detach().squeeze(0)  # 记录初始超分辨率图片sr0
        lr = lr.squeeze(0)
        lr0 = lr  # 记录初始低分辨图片lr0
        lr = lr.unsqueeze(0)

        dim = [ i for i in range(0, part_size * part_size )]
        start_pos_x = int(lr0.size()[1] / 3)
        start_pos_y = int(lr0.size()[2] / 3)
        last_loss = 0

        for temT in range(T):
            shuffle(dim)
            for j in range(0, part_size * part_size):
                zero_lr = torch.zeros_like(lr0)
                tem_dim_x = math.floor(dim[j] / part_size)
                tem_dim_y = dim[j] % part_size
                zero_lr[:, start_pos_x + tem_dim_x, start_pos_y + tem_dim_y] = 1
                lr_new = torch.clamp(lr + zero_lr.unsqueeze(0) * alpha / T, 0, 1)
                sr = net(lr_new, cfg.scale).detach().squeeze(0)

                if cfg.picture_quality == 'L2':
                    loss = torch.sqrt(loss_fun(sr, sr0) * lr.numel())
                else:
                    loss = 1 - pytorch_ssim.ssim(sr.unsqueeze(0), sr0.unsqueeze(0))
                lr_new2 = torch.clamp(lr - zero_lr.unsqueeze(0) * alpha / T, 0, 1)
                sr2 = net(lr_new2, cfg.scale).detach().squeeze(0)
                if cfg.picture_quality == 'L2':
                    loss2 = torch.sqrt(loss_fun(sr2, sr0) * lr.numel())
                else:
                    loss2 = 1 - pytorch_ssim.ssim(sr.unsqueeze(0), sr0.unsqueeze(0))
                if last_loss > max(loss, loss2):
                    continue
                if loss > loss2:
                    lr = lr_new
                else:
                    lr = lr_new2
                last_loss = max(loss, loss2)

            if temT == 0:
                print('第一次迭代后loss: {:.3f}'.format(max(loss, loss2)))

        print('最终loss: {:.3f}'.format(max(loss, loss2)))
        print('SSIM_SR: {:.3f}'.format(pytorch_ssim.ssim(sr.unsqueeze(0), sr0.unsqueeze(0))))
        psnr_sr = cal_psnr(sr, sr0)
        psnr_lr = cal_psnr(lr.squeeze(0), lr0)
        print('PSNR_LR: {:.3f}'.format(psnr_lr))
        print('PSNR_SR: {:.3f}'.format(psnr_sr))

        save_pic(cfg, name, lr.squeeze(0), lr0, sr, sr0)


def attack(net, device, dataset, cfg, alpha, T):
    loss_fun = nn.MSELoss()

    for step, (hr, lr, name) in enumerate(dataset):
        # print(lr.size())
        # print(lr.numel())
        print('=====================图片:{}====================='.format(name))
        hr = hr.to(device)
        lr = lr.unsqueeze(0).to(device)
        sr0 = net(lr, cfg.scale).detach().squeeze(0)  # 记录初始超分辨率图片sr0
        lr = lr.squeeze(0)
        lr0 = lr   # 记录初始低分辨图片lr0

        mask = torch.zeros_like(lr0)
        mask_x = int(mask.size()[1] / 3)
        mask_y = int(mask.size()[2] / 3)
        mask[:, mask_x:mask_x+cfg.part_size, mask_y:mask_y+cfg.part_size] = 1

        sr_mask = torch.zeros_like(sr0)
        sr_mask[:,  4*mask_x:4*(mask_x + cfg.part_size), 4*mask_y:4*(mask_y + cfg.part_size)] = 1
        outer_mask = (torch.ones_like(sr0) - sr_mask)

        lr_rand = torch.Tensor(np.random.uniform(-0.2 / 255, 0.2 / 255, lr.shape)).type_as(lr).to(device)
        if cfg.type == 'part':
            lr = lr + lr_rand * mask   # lr0加随机偏移，否则梯度为0，无法计算
        if cfg.type == 'white':
            lr = lr + lr_rand

        lr = lr.unsqueeze(0)
        lr.requires_grad = True

        for temT in range(T):
            # print(lr)
            sr = net(lr, cfg.scale).squeeze(0)
            net.zero_grad()
            '''
            if (cfg.type == 'part'):
                sr = sr * outer_mask
            '''
            if cfg.picture_quality == 'L2':
                loss = torch.sqrt(loss_fun(sr, sr0) * sr.numel())  # L2距离
            else:
                loss = 1 - pytorch_ssim.ssim(sr.unsqueeze(0), sr0.unsqueeze(0))
            if temT == 0:
                print('start loss: {:.3f}'.format(loss.data))
            loss.backward()
            data_grad = lr.grad.data
            # print(data_grad)
            lr = lr.detach()
            lrn = torch.clamp(lr + torch.sign(data_grad) * alpha / T, 0, 1)
            # print(lrn)
            if cfg.type == 'part':
                lr = (torch.clamp((lrn - lr0), -alpha, alpha)) * mask + lr0
            else:
                lr = (torch.clamp((lrn - lr0), -alpha, alpha)) + lr0
            lr = Variable(lr)
            lr.requires_grad = True

        # 迭代后结果:低分辨图lr，超分辨图sr
        sr = net(lr, cfg.scale).squeeze(0)
        loss = torch.sqrt(loss_fun(sr, sr0) * lr.numel())
        print('final loss: {:.3f}'.format(loss.data))
        print('SSIM_SR: {:.3f}'.format(pytorch_ssim.ssim(sr.unsqueeze(0), sr0.unsqueeze(0))))
        psnr_sr = cal_psnr(sr, sr0)
        psnr_lr = cal_psnr(lr.squeeze(0), lr0)
        if step == 0:
            PSNR_LR.append(psnr_lr)
            PSNR_SR.append(psnr_sr)
        print('PSNR_LR: {:.3f}'.format(psnr_lr) )
        print('PSNR_SR: {:.3f}'.format(psnr_sr))
        sr = net(lr, cfg.scale).detach().squeeze(0)
        lr = lr.squeeze(0)

        save_pic(cfg, name, lr, lr0, sr, sr0)


def draw():
    pyplot.title("攻击效果", fontsize=30)
    pyplot.xlabel("像素值变化范围", fontsize=20)
    pyplot.ylabel("PSNR", fontsize=20)
    x = range(0, 6, 1)
    A, = pyplot.plot(PSNR_LR, "r-o")
    B, = pyplot.plot(PSNR_SR, "g-o")
    pyplot.legend([A, B], ["PSNR_LR", "PSNR_SR"], fontsize=16)
    pyplot.xticks(x, ('1', '2', '4', '8', '16', '32'), fontsize=16)
    pyplot.yticks(fontsize=16)
    pyplot.show()


def cal_psnr(pic1, pic2):
    psnr = 0
    func = nn.MSELoss()
    for i in range(3):
        x = 255 * pic1[i, :, :]
        y = 255 * pic2[i, :, :]
        mse = func(x, y)
        if mse == 0:
            return 100
        psnr += 10 * math.log10(255 * 255 / mse)
    psnr = psnr / 3
    return psnr


def save_pic(cfg, name, lr, lr0,sr, sr0):
    model_name = cfg.ckpt_path.split(".")[2].split("/")[-1]

    dir = os.path.join(cfg.sample_dir,
                       model_name,
                       cfg.test_data_dir.split("/")[-1],
                       "x{}".format(cfg.scale))

    os.makedirs(dir, exist_ok=True)
    lr_im_path = os.path.join(dir, "{}".format(name))
    lr0_im_path = os.path.join(dir, "{}".format(name.split(".")[0][:-2] + "LR0" + '.png'))
    sr_im_path = os.path.join(dir, "{}".format(name.split(".")[0][:-2] + "SR" + '.png'))
    sr0_im_path = os.path.join(dir, "{}".format(name.split(".")[0][:-2] + "SR0" + '.png'))

    save_image(sr, sr_im_path)
    save_image(sr0, sr0_im_path)
    save_image(lr, lr_im_path)
    save_image(lr0, lr0_im_path)


def main(cfg):
    net = Net(multi_scale=True, group=cfg.group, scale=4)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    state_dict = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    dataset = TestDataset(cfg.test_data_dir, cfg.scale)
    if cfg.type == 'black':
        black_attack(net, device, dataset, cfg, cfg.alpha, cfg.T)
    else:
        attack(net, device, dataset, cfg, cfg.alpha, cfg.T)


if __name__ == "__main__":
    cfg = attack_parse_args()
    main(cfg)

