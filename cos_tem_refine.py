# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Peiwen Lin, Xiangtai Li
# Email: linpeiwen@sensetime.com, lixiangtai@sensetime.com

from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import module
from pss.utils.bn_helper import get_syncbn
from pss.models.base import ASPP, Aux_Module


class dec_cos_tem(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, num_classes=19, inner_planes=512,
                 sync_bn=False, dilations=(12, 24, 36), with_aux=True, criterion=None):
        super(dec_cos_tem, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.with_aux = with_aux
        self.stl = STL(512+256)
        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        if self.with_aux:
            self.aux_layer = Aux_Module(in_planes//2, num_classes, norm_layer)
        self.criterion = criterion

    def forward(self, x):
        x1, x2, x3, x4 = x['features']
        x1 = nn.AdaptiveAvgPool2d(x4.shape[2:])(x1)
        f = torch.cat([x2, x1], dim=1)
        f = self.stl(f)
        size = x['size']
        aspp_out = self.aspp(x4)
        aspp_out = torch.cat([aspp_out, f], dim=1)
        pred = self.head(aspp_out)

        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training and self.with_aux:
            gt_seg = x['gt_seg']
            aux_pred = self.aux_layer(x3)
            aux_pred = F.upsample(aux_pred, size=size, mode='bilinear', align_corners=True)
            pred = pred, aux_pred
            return {"loss": self.criterion(pred, gt_seg), "blob_pred": pred[0]}
        else:
            return {"blob_pred": pred}


class dec_deeplabv3p(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al.. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"*
    """
    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), criterion=None):
        super(dec_deeplabv3p, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)

        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes() + 48, 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.fine = nn.Conv2d(256, 48, 1)

        if self.with_aux:
            self.aux_layer = Aux_Module(in_planes//2, num_classes, norm_layer)
        self.criterion = criterion

    def forward(self, x):
        x1, x2, x3, x4 = x['features']
        size = x['size']
        aspp_out = self.aspp(x4)
        bot = self.fine(x1)
        out = torch.cat([aspp_out, bot],dim=1)
        pred = self.head(out)
        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)
        if self.training and self.with_aux:
            gt_seg = x['gt_seg']
            aux_pred = self.aux_layer(x3)
            aux_pred = F.upsample(aux_pred, size=size, mode='bilinear', align_corners=True)
            pred = pred, aux_pred
            return {"loss": self.criterion(pred, gt_seg), "blob_pred": pred[0]}
        else:
            return {"blob_pred": pred}

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                    has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        norm_layer = get_syncbn()
        '''
        if mode == '1d':
            norm_layer = nn.BatchNorm1d
        elif mode == '2d':
            norm_layer = nn.BatchNorm2d
        '''
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, bias=False, groups=group)
        elif mode == '1d':
            self.conv = nn.Conv1d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride, 
                    padding=padding, dilation=dilation, bias=False, groups=group)
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class QCO_1d(nn.Module):
    def __init__(self, level_num):
        super(QCO_1d, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape

        x_ave = F.adaptive_avg_pool2d(x, (1, 1)) #[N, 128, 1, 1]
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1) #[N, H, W]
        cos_sim = cos_sim.view(N, -1) #[N, HW]

        cos_sim_min, _ = cos_sim.min(-1) #[N]
        cos_sim_min = cos_sim_min.unsqueeze(-1) #[N, 1]
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1) #[N, 1]

        q_levels = torch.arange(self.level_num).float().cuda() #[s]
        q_levels = q_levels.expand(N, self.level_num) #[N, s]


        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min #[N, s]
        q_levels = q_levels.unsqueeze(1) #[N, 1, s] 
        #q_levels_spatial = q_levels.expand(H*W, N, self.level_num).permute(1, 0, 2) #[N, HW, s]

        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] #[N, 1]
        q_levels_inter = q_levels_inter.unsqueeze(-1) #[N, 1, 1]

        cos_sim = cos_sim.unsqueeze(-1) #[N, HW, 1]


        quant = 1 - torch.abs(q_levels - cos_sim)
        quant = quant * (quant > (1 - q_levels_inter)) #[N, HW, s]

        sta = quant.sum(1) #[N, s]
        sta = sta / (sta.sum(-1).unsqueeze(-1)) #[N, s]
        sta = sta.unsqueeze(1) #[N, 1, s]

        sta = torch.cat([q_levels, sta], dim=1) #[N, 2, s]
        sta = self.f1(sta)
        sta = self.f2(sta) #[N, C, S]
        
        x_ave = x_ave.squeeze(-1).squeeze(-1) #[N, C]
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0) #[N, C, S]
        
        sta = torch.cat([sta, x_ave], dim=1)
        sta = self.out(sta)

        return sta, quant

class QCO_2d(nn.Module):
    def __init__(self, scale, level_num):
        super(QCO_2d, self).__init__()
        self.f1 = nn.Sequential(ConvBNReLU(3, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='2d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='2d')
        self.out = nn.Sequential(ConvBNReLU(256+128, 128, 1, 1, 0, has_bn=True, has_relu=True, mode='2d'), ConvBNReLU(128, 128, 1, 1, 0, has_bn=True, has_relu=False, mode='2d'))
        self.scale = scale
        self.level_num = level_num
    def forward(self, x):
        N1, C1, H1, W1 = x.shape
        if H1 // self.level_num != 0 or W1 // self.level_num != 0:
            x = F.adaptive_avg_pool2d(x, ((int(H1/self.level_num)*self.level_num), int(W1/self.level_num)*self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = F.adaptive_avg_pool2d(x, (self.scale, self.scale)) #[N, C, s, s]
        x_ave_up = F.adaptive_avg_pool2d(x_ave, (H, W)) #[N, C, H, W]
        cos_sim = (F.normalize(x_ave_up, dim=1) * F.normalize(x, dim=1)).sum(1) #[N, H, W]
        cos_sim = cos_sim.unsqueeze(1) #[N, 1, H, W]
        #cos_sim = F.pad(cos_sim, (1, 1, 1, 1), mode='constant', value=0.) #[N, 1, H+1, W+1]
        #cos_sim = torch.cat([cos_sim[:, :, 1:-1, 1:-1], cos_sim[:, :, 1:-1, 2:]], dim=1) #[N, 2, H, W]

        cos_sim = cos_sim.reshape(N, 1, self.scale, self.size_h, self.scale, self.size_w)
        cos_sim = cos_sim.permute(0, 1, 2, 4, 3, 5)
        cos_sim = cos_sim.reshape(N, 1, int(self.scale*self.scale), int(self.size_h*self.size_w)) #[N, 2, s*s, h*w]
        cos_sim = cos_sim.permute(0, 1, 3, 2) #[N, 1, h*w, s*s]
        cos_sim = cos_sim.squeeze(1) #[N, h*w, s*s]

        cos_sim_min, _ = cos_sim.min(1) #[N, s*s]
        cos_sim_min = cos_sim_min.unsqueeze(-1) #[N, s*s, 1]
        cos_sim_max, _ = cos_sim.max(1) #[N, s*s]
        cos_sim_max = cos_sim_max.unsqueeze(-1) #[N, s*s, 1]

        q_levels = torch.arange(self.level_num).float().cuda() #[l]
        q_levels = q_levels.expand(N, self.scale*self.scale, self.level_num) #[N, s*s, l]
        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min #[N, s*s, l]
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] #[N, s*s]
        q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1) #[N, 1, s*s, 1]

        cos_sim = cos_sim.unsqueeze(-1) #[N, h*w, s*s, 1]
        q_levels = q_levels.unsqueeze(1)#[N, 1, s*s, l]

        quant = 1 - torch.abs(q_levels - cos_sim) #[N, h*w, s*s, l]
        quant = quant * (quant > (1 - q_levels_inter)) #[N, h*w, s*s, l]

        quant = quant.view([N, self.size_h, self.size_w, self.scale*self.scale, self.level_num]) #[N, h, w, s*s, l]
        quant = quant.permute(0, -2, -1, 1, 2) #[N, s*s, l, h, w]


        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3) #[N, s*s, l, 1, h, w]
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2) #[N, s*s, 1, l, h, w]
        quant = quant_left * quant_right #[N, s*s, l, l, h, w]

        sta = quant.sum(-1).sum(-1) #[N, s*s, l, l]
        sta = sta / (sta.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)) #[N, s*s, l, l]
        sta = sta.unsqueeze(1) #[N, 1, s*s, l, l]
        
        q_levels = q_levels.expand(self.level_num, N, 1, self.scale*self.scale, self.level_num)
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4) #[N, 1, s*s, l, l]
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3) #[N, 1, s*s, l, l]
        sta = torch.cat([q_levels_h, q_levels_w, sta], dim=1) #[N, 3, s*s, l, l]
        sta = sta.view(N, 3, self.scale * self.scale, -1) #[N, 3, s*s, l*l]
        
        sta = self.f1(sta)
        sta = self.f2(sta) #[N, C, s*s, l*l]

        x_ave = x_ave.view(N, C, -1) #[N, C, s*s]
        x_ave = x_ave.expand(self.level_num*self.level_num, N, C, self.scale*self.scale)
        x_ave = x_ave.permute(1, 2, 3, 0) #[N, C, s*s, l*l]

        sta = torch.cat([x_ave, sta], dim=1)
        sta = self.out(sta) #[N, C, s*s, l*l]
        sta = sta.mean(-1)
        sta = sta.view(N, -1, self.scale, self.scale)
        return sta



class TEM(nn.Module):
    def __init__(self, level_num):
        super(TEM, self).__init__()
        self.level_num = level_num
        self.qco = QCO_1d(level_num)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')
    def forward(self, x): #quant [N, HW, S]
        N, C, H, W = x.shape
        sta, quant = self.qco(x) #sta [N, C, S]  quant [N, HW, S]
        k = self.k(sta) #[N, C, S]
        q = self.q(sta) #[N, C, S]
        v = self.v(sta) #[N, C, S]
        k = k.permute(0, 2, 1) #[N, S, C]
        w = torch.bmm(k, q) #[N, S, S]
        w = F.softmax(w, dim=-1) #[N, S, S]
        v = v.permute(0, 2, 1) #[N, S, C]
        f = torch.bmm(w, v) #[N, S, C]
        f = f.permute(0, 2, 1) #[N, C, S]
        f = self.out(f) #[N, C, S]
        quant = quant.permute(0, 2, 1) #[N, S, HW]
        out = torch.bmm(f, quant) #[N, C, HW]
        out = out.view(N, 256, H, W)
        return out


class PTFEM(nn.Module):
    def __init__(self):
        super(PTFEM, self).__init__()
        self.conv = ConvBNReLU(512, 256, 1, 1, 0, has_bn=False, has_relu=False)
        self.qco_1 = QCO_2d(1, 8)
        self.qco_2 = QCO_2d(2, 8)
        self.qco_3 = QCO_2d(3, 8)
        self.qco_6 = QCO_2d(6, 8)
        self.out = ConvBNReLU(512, 256, 1, 1, 0)
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.conv(x)
        sta_1 = self.qco_1(x) #[N, C, s*s, l*l]
        sta_2 = self.qco_2(x)
        sta_3 = self.qco_3(x)
        sta_6 = self.qco_6(x)
        
        N, C = sta_1.shape[:2]

        sta_1 = sta_1.view(N, C, 1, 1)
        sta_2 = sta_2.view(N, C, 2, 2)
        sta_3 = sta_3.view(N, C, 3, 3)
        sta_6 = sta_6.view(N, C, 6, 6)

        sta_1 = F.interpolate(sta_1, size=(H, W), mode='bilinear', align_corners=True)
        sta_2 = F.interpolate(sta_2, size=(H, W), mode='bilinear', align_corners=True)
        sta_3 = F.interpolate(sta_3, size=(H, W), mode='bilinear', align_corners=True)
        sta_6 = F.interpolate(sta_6, size=(H, W), mode='bilinear', align_corners=True)

        x = torch.cat([sta_1, sta_2, sta_3, sta_6], dim=1) #512
        x = self.out(x)
        return x



class T_conv_pyramid(nn.Module):
    def __init__(self, in_num):
        super().__init__()
        self.conv_start = ConvBNReLU(in_num, 256, 1, 1, 0)
        '''
        self.p1 = T_conv(16, 128, 256, 256, 256)
        self.p2 = T_conv(8, 128, 256, 256, 256)
        self.p3 = T_conv(4, 128, 256, 256, 256)
        '''
        self.p4 = TEM(128)

    def forward(self, x):
        x = self.conv_start(x)
        x_4 = self.p4(x)
        x = torch.cat([x, x_4], dim=1)
        return x

class STL(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_start = ConvBNReLU(in_channel, 256, 1, 1, 0)
        self.tem = TEM(128)
        self.ptfem = PTFEM()
    def forward(self, x):
        x = self.conv_start(x)
        x_tem = self.tem(x)
        x = torch.cat([x_tem, x], dim=1) #c = 256 + 256 = 512
        x_ptfem = self.ptfem(x) # 256   
        x = torch.cat([x_ptfem, x], dim=1)
        return x



