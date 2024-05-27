#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jinglei Shi
@file: depth_estimator.py
@time: 08/04/2020 13:06
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _bwd2d

class _Feature_extractor(nn.Module):
    def __init__(self):
        super(_Feature_extractor,self).__init__()
        self.conv1 = self.conv(3,16)
        self.conv2 = self.conv(16,32)
        self.conv3 = self.conv(32,64)
        self.conv4 = self.conv(64,96)
        self.conv5 = self.conv(96,128)
        self.conv6 = self.conv(128,196)
    def conv(self,feat_in,feat_out):
        return nn.Sequential(
            nn.Conv2d(feat_in,feat_out,3,2,1),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(feat_out,feat_out,3,1,1),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(feat_out,feat_out,3,1,1),
            nn.LeakyReLU(0.1,True)
        )
    def forward(self,x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)
        feat6 = self.conv6(feat5)
        return [feat2,feat3,feat4,feat5,feat6]

class _Flow_estimator(nn.Module):
    def __init__(self,feat_in):
        super(_Flow_estimator,self).__init__()
        self.conv1 = nn.Conv2d(feat_in,128,3,1,1)
        self.conv2 = nn.Conv2d(128+feat_in,128,3,1,1)
        self.conv3 = nn.Conv2d(128+128+feat_in,96,3,1,1)
        self.conv4 = nn.Conv2d(96+128+128+feat_in,64,3,1,1)
        self.conv5 = nn.Conv2d(64+96+128+128+feat_in,32,3,1,1)
        self.conv6 = nn.Conv2d(32+64+96+128+128+feat_in,2,3,1,1)
        self.lru = nn.LeakyReLU(0.1,True)
    def forward(self,x):
        conv1 = self.lru(self.conv1(x))
        x = torch.cat([conv1,x],1)
        conv2 = self.lru(self.conv2(x))
        x = torch.cat([conv2,x],1)
        conv3 = self.lru(self.conv3(x))
        x = torch.cat([conv3,x],1)
        conv4 = self.lru(self.conv4(x))
        x = torch.cat([conv4,x],1)
        conv5 = self.lru(self.conv5(x))
        x = torch.cat([conv5,x],1)
        flow = self.conv6(x)
        return [flow,x]

class _Context_refinement(nn.Module):
    def __init__(self):
        super(_Context_refinement,self).__init__()
        self.conv1 = nn.Conv2d(565,128,3,1,1,1)
        self.conv2 = nn.Conv2d(128,128,3,1,2,2)
        self.conv3 = nn.Conv2d(128,128,3,1,4,4)
        self.conv4 = nn.Conv2d(128,96,3,1,8,8)
        self.conv5 = nn.Conv2d(96,64,3,1,16,16)
        self.conv6 = nn.Conv2d(64,32,3,1,1)
        self.residue = nn.Conv2d(32,2,3,1,1)
        self.lru = nn.LeakyReLU(0.1,True)
    def forward(self,x):
        x = self.lru(self.conv1(x))
        x = self.lru(self.conv2(x))
        x = self.lru(self.conv3(x))
        x = self.lru(self.conv4(x))
        x = self.lru(self.conv5(x))
        x = self.lru(self.conv6(x))
        return self.residue(x)

class _Upsampler(nn.Module):
    def __init__(self,feat_in):
        super(_Upsampler, self).__init__()
        self.up_flow = nn.ConvTranspose2d(2,2,4,2,1)
        self.up_feat = nn.ConvTranspose2d(feat_in,2,4,2,1)
    def forward(self, flow,feat):
        return [self.up_flow(flow),self.up_feat(feat)]

class PWC_net(nn.Module):
    def __init__(self,scale=2):
        super(PWC_net, self).__init__()
        self.sr = 4
        self.scale = scale
        self.corr_num = (self.sr*2+1)**2
        self.lru = nn.LeakyReLU(0.1,True)

        self.tensor_up = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=True)
        self.tensor_down = nn.Upsample(scale_factor=4/self.scale,mode='bilinear',align_corners=True)

        self._feature_extractor = _Feature_extractor()
        self._context_network = _Context_refinement()

        self._flow_estimator5 = _Flow_estimator(self.corr_num)
        self._upsampler5 = _Upsampler(32+64+96+128+128+self.corr_num)

        self._flow_estimator4 = _Flow_estimator(self.corr_num+128+2+2)
        self._upsampler4 = _Upsampler(32+64+96+128+128+self.corr_num+128+2+2)

        self._flow_estimator3 = _Flow_estimator(self.corr_num+96+2+2)
        self._upsampler3 = _Upsampler(32+64+96+128+128+self.corr_num+96+2+2)

        self._flow_estimator2 = _Flow_estimator(self.corr_num+64+2+2)
        self._upsampler2 = _Upsampler(32+64+96+128+128+self.corr_num+64+2+2)

        self._flow_estimator1 =_Flow_estimator(self.corr_num+32+2+2)

    def _corr2d(self,x,y,max_disp=4):
        corr_tensor = []
        B,C,H,W = y.size()
        p2d = [max_disp,max_disp,max_disp,max_disp]
        pad_y = F.pad(y,p2d,'constant',0)
        for i in range(-max_disp,max_disp+1,1):
            for j in range(-max_disp,max_disp+1,1):
                slice_y = pad_y[:,:,max_disp+i:(max_disp+i+H),max_disp+j:(max_disp+j+W)]
                cost = torch.mean(x*slice_y,1)
                corr_tensor.append(cost)
        return torch.stack(corr_tensor,1)

    def forward(self,input_a,input_b):

        up_a = self.tensor_up(input_a)
        up_b = self.tensor_up(input_b)

        [feat1a,feat2a,feat3a,feat4a,feat5a] = self._feature_extractor(up_a)
        [feat1b,feat2b,feat3b,feat4b,feat5b] = self._feature_extractor(up_b)

        corr5 = self.lru(self._corr2d(feat5a,feat5b))
        pred_flow5,feat5 = self._flow_estimator5(corr5)
        up_flow5,up_feat5 = self._upsampler5(pred_flow5,feat5)

        warp4 = _bwd2d(feat4b, up_flow5 * 0.625)
        corr4 = self.lru(self._corr2d(feat4a,warp4))
        pred_flow4,feat4 = self._flow_estimator4(torch.cat([corr4,feat4a,up_flow5,up_feat5],1))
        up_flow4,up_feat4 = self._upsampler4(pred_flow4,feat4)

        warp3 = _bwd2d(feat3b, up_flow4 * 1.25)
        corr3 = self.lru(self._corr2d(feat3a,warp3))
        pred_flow3,feat3 = self._flow_estimator3(torch.cat([corr3,feat3a,up_flow4,up_feat4],1))
        up_flow3,up_feat3 = self._upsampler3(pred_flow3,feat3)

        warp2 = _bwd2d(feat2b, up_flow3 * 2.5)
        corr2 = self.lru(self._corr2d(feat2a,warp2))
        pred_flow2,feat2 = self._flow_estimator2(torch.cat([corr2,feat2a,up_flow3,up_feat3],1))
        up_flow2,up_feat2 = self._upsampler2(pred_flow2,feat2)

        warp1 = _bwd2d(feat1b, up_flow2 * 5.0)
        corr1 = self.lru(self._corr2d(feat1a,warp1))
        pred_flow1,feat1 = self._flow_estimator1(torch.cat([corr1,feat1a,up_flow2,up_feat2],1))

        pred_flow = pred_flow1 + self._context_network(feat1)

        flow = self.tensor_down(pred_flow*20/self.scale)
        return flow
