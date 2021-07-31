import pickle

import torch
import torch.nn as nn
import torch.nn.init as init


import math

from utils.modules.common import *
from utils.modules.lw_net_Ex import MODEL



class LinearSISR(nn.Module):
    def __init__(self,config):
        super(LinearSISR, self).__init__()
        self.config = config

        self.weightReg =  MODEL(config.MODEL)

        # kernels =pickle.load(open(config.MODEL.KERNEL_PATH, 'rb')).reshape((config.MODEL.NWEIGT,-1))
        kernels = np.load(config.MODEL.KERNEL_PATH).reshape((config.MODEL.NWEIGT,-1))
        kernels = torch.from_numpy(kernels).float().cuda()

        self.kernelSize = int(math.sqrt(kernels.shape[-1]))
        self.register_parameter('K',torch.nn.Parameter(kernels))
        print('self.K',self.K.shape)
        self.K.requires_grad = config.MODEL.fineturningK
        self.s = config.MODEL.SCALE

        self.criterion = nn.L1Loss(reduction='mean')

        self.pointConv = torch.nn.Conv2d(config.MODEL.NWEIGT,config.MODEL.NWEIGT,3,1,1)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    def convFilters(self,kernels,x):
        # print('x',x.shape)
        bicubic_x = torch.nn.functional.interpolate(x,scale_factor=2,mode = 'bicubic',align_corners=False)

        b,c,h,w = bicubic_x.shape
        padSize = (self.kernelSize-1)//2
        pad_bicubic = torch.nn.functional.pad(bicubic_x,pad=(padSize,padSize,padSize,padSize),mode="reflect")
        pad_bicubic_unfold = torch.nn.functional.unfold(pad_bicubic ,self.kernelSize,\
                    padding = 0).view(b,c,self.kernelSize*self.kernelSize,-1).permute(0,1,3,2).contiguous()
        pad_bicubic_unfold = pad_bicubic_unfold.view(b,c ,h,w,-1)

        print(pad_bicubic_unfold.shape,kernels.shape)

        output = torch.einsum('bchwd,bhwd->bchw',pad_bicubic_unfold,kernels)
        return output
    def forward(self, x, gt=None):
        # print(x.shape)
        
        reg_weights = self.weightReg(x)

        reg_weights = torch.nn.functional.interpolate(reg_weights,scale_factor=2,mode='bilinear',align_corners=False)

        
        mix_kernel = torch.einsum('bhwd,dq->bhwq',reg_weights.permute(0,2,3,1),self.K)
        

        out = self.convFilters(mix_kernel,x)
        if gt is not None:
            l1_loss = self.criterion(out, gt) 
            return dict(L1=l1_loss)
        else:
            return out

    def loadWeights(self,modelPath):
        preTrained = torch.load(modelPath,map_location='cpu')
        self.load_state_dict(preTrained)

if __name__ == '__main__':
    from config import config
    net = LinearSISR(config).cuda()
    # print('net',net)
    print("backbone have {:.3f}M paramerters in total".format(sum(x.numel() for x in net.parameters())/1000000.0))
    ins = torch.randn(1, 3, 64, 64).cuda()
    output = net(ins)
    print('output',output.size())
    input('check')

