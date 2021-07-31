import cv2
import os

import torch

from utils.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr

import numpy as np
def validate(model, val_loader, device, iteration, down=4, to_y=True, save_path='.', save_img=False, max_num=10):
    # for batch=1
    psnr_l = []
    ssim_l = []
    for idx, (lr_img, hr_img) in enumerate(val_loader):
        if idx >= max_num:
            break

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        h, w = lr_img.size()[2:]
        need_pad = False
        if h % down != 0 or w % down != 0:
            need_pad = True
            pad_y_t = (down - h % down) % down // 2
            pad_y_b = (down - h % down) % down - pad_y_t
            pad_x_l = (down - w % down) % down // 2
            pad_x_r = (down - w % down) % down - pad_x_l
            lr_img = torch.nn.functional.pad(lr_img, pad=(pad_x_l, pad_x_r, pad_y_t, pad_y_b), mode='replicate')

        with torch.no_grad():
            output = model(lr_img)

        if need_pad:
            scale = output.size(2) // lr_img.size(2)
            y_end = -pad_y_b * scale if pad_y_b != 0 else output.size(2)
            x_end = -pad_x_r * scale if pad_x_r != 0 else output.size(3)
            output = output[:, :, pad_y_t * scale: y_end, pad_x_l * scale: x_end]

        output = tensor2img(output).astype(np.float32)/255.0
        gt = tensor2img(hr_img).astype(np.float32)/255.0

        if save_img:
            ipath = os.path.join(save_path, '%d_%d.png' % (iteration, idx))
            cv2.imwrite(ipath, output)

        if to_y:
            output = bgr2ycbcr(output, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            
        output = output[2:-2,2:-2]
        gt = gt[2:-2,2:-2]
        psnr = calculate_psnr(output*255.0, gt*255.0)
        ssim = calculate_ssim(output*255.0, gt*255.0)
        psnr_l.append(psnr)
        ssim_l.append(ssim)

    avg_psnr = sum(psnr_l) / len(psnr_l)
    avg_ssim = sum(ssim_l) / len(ssim_l)

    return avg_psnr, avg_ssim


if __name__ == '__main__':
    from config_local import config
    from network import LinearSISR
    from dataset import get_val_dataset
    from utils import dataloader
    from utils.model_opr import load_model
    from utils.common import *
    from metric.metric import evaluationPSNR
    model = LinearSISR(config)
    device = torch.device('cuda')
    model = model.to(device)
    model_path = './out/200000.pth'
    # model.loadWeights()
    # load_model(model, model_path)

    # val_dataset = get_val_dataset(config)
    # val_loader = dataloader.val_loader(val_dataset, config, 0, 1)
    # print(validate(model, val_loader, device, 40000, down=4, to_y=True, save_img=False))
    import glob
    model.eval()
    print("backbone have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))

    keyWord = 'B100'
    scale = 2
    bankchmark = '/home/redrock/project/DRN-master/benchmark'
    rootPath = os.path.join(bankchmark,keyWord)

    hrPath = os.path.join(rootPath,'HR')
    lrPath = os.path.join(rootPath,'LR_bicubic','X{}'.format(scale))

    print('hrPath',hrPath,os.path.exists(hrPath),lrPath,os.path.exists(lrPath))

    hr_images = sorted(glob.glob(hrPath+'/*.png'))
    lr_images = sorted(glob.glob(lrPath+'/*x{}.png'.format(scale)))

    writePath = '{}/{}'.format('/home/redrock/project/sisr_drn/adakernel/exps/baseline_zk/out/benchmark',keyWord)
    if os.path.exists(writePath) is False:
        os.mkdir(writePath)

    print('hr_images',hr_images,lr_images)
    p_sum,pb_sum = 0,0
    count = 0
    with torch.no_grad():
        for lr_fp,hr_fp in zip(lr_images, hr_images):
            keyName = lr_fp.split('/')[-1].split('.')[0][:-2]
            print('keyName',keyName)

            hr = cv2.imread(hr_fp)[:,:,::-1]
            lr = cv2.imread(lr_fp)[:,:,::-1]

            h = int(lr.shape[0]/4)*4
            w = int(lr.shape[1]/4)*4
            lr = lr[:h,:w]

            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
            cv2.imshow('hr',hr)
            cv2.imshow('lr',lr)
            cv2.waitKey(20)

            lr_tensor,hr_tensor = imgToTensor(lr),imgToTensor(hr)
            bicubic = torch.nn.functional.interpolate(lr_tensor,scale_factor=2,mode='bicubic',align_corners=False)
            psnr,psnr_bicubic,out = evaluationPSNR(model,lr_tensor,hr_tensor,getOut) # net,ins,label,out_func,hr=None
            print('psnr,psnr_bicubic',psnr,psnr_bicubic,out.shape)
            # input('check')

            img_cons = out.detach().cpu().numpy()[0]
            print('img_cons',img_cons.max(),img_cons.min(),img_cons.shape)
            img_cons = np.transpose(img_cons,(1,2,0))*255.0
            img_cons = np.clip(img_cons,0,255).astype(np.uint8)[:,:,::-1]
            cv2.imshow('img_cons',img_cons)

            img_bic= bicubic.cpu().numpy()[0]
            print('img_bic',img_bic.max(),img_bic.min(),img_bic.shape)
            img_bic = np.transpose(img_bic,(1,2,0))*255.0
            img_bic = np.clip(img_bic,0,255).astype(np.uint8)[:,:,::-1]
            cv2.imshow('img_bic',img_bic)


            cv2.imwrite(writePath+'/{}ex.png'.format(keyName),img_cons)
            cv2.imwrite(writePath+'/{}bic.png'.format(keyName),img_bic)
            cv2.imwrite(writePath+'/{}hr.png'.format(keyName),hr[:,:,::-1])
            cv2.waitKey(20)

            p_sum += psnr 
            pb_sum += psnr_bicubic
            count +=1
        print(p_sum/count,pb_sum/count)