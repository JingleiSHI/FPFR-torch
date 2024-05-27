#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jinglei Shi
@file: FPFR.py
@time: 08/04/2020 13:05
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.depth_estimator import PWC_net
from utils import _bwd2d,_fwd2d
import torchvision.models as models

class _Pix_net(nn.Module):
    def __init__(self):
        super(_Pix_net, self).__init__()
        self.conv1 = nn.Conv2d(16,64,3,1,1)
        self.conv2 = nn.Conv2d(64,32,3,1,1)
        self.conv3 = nn.Conv2d(32,16,3,1,1)
        self.conv_pix = nn.Conv2d(16,3,3,1,1)
        self.conv_confid = nn.Conv2d(16,8,3,1,1)
        self.lru = nn.LeakyReLU(0.1,True)

    def forward(self, x):
        x = self.lru(self.conv1(x))
        x = self.lru(self.conv2(x))
        x = self.lru(self.conv3(x))
        img_pix = self.conv_pix(x)
        confid_pix = self.lru(self.conv_confid(x))
        return img_pix,confid_pix

class _Feat_net(nn.Module):
    def __init__(self):
        super(_Feat_net, self).__init__()
        self.decoder3 = self.conv(256*4+4,256)
        self.decoder2 = self.conv(128*4+32+4,128)
        self.decoder1 = self.conv(64*4+16+4,64)
        self.up_sampler3 = nn.ConvTranspose2d(256,32,4,2,1)
        self.up_sampler2 = nn.ConvTranspose2d(128,16,4,2,1)
        self.decoder = nn.Sequential(
            nn.Conv2d(64,32,3,1,1), nn.LeakyReLU(0.1,True),
            nn.Conv2d(32,32,3,1,1), nn.LeakyReLU(0.1,True)
        )
        self.conv_img = nn.Conv2d(32,3,3,1,1)
        self.conv_confid = nn.Conv2d(32,8,3,1,1)
        self.lru = nn.LeakyReLU(0.1,True)

    def conv(self,feat_in,feat_out):
        conv1 = nn.Conv2d(feat_in,feat_out,3,1,1)
        conv2 = nn.Conv2d(feat_out,feat_out,3,1,1)
        conv3 = nn.Conv2d(feat_out,feat_out,3,1,1)
        decode_block = nn.Sequential(
            conv1, nn.LeakyReLU(0.1,True),
            conv2, nn.LeakyReLU(0.1,True),
            conv3, nn.LeakyReLU(0.1,True)
        )
        return decode_block

    def forward(self, feat_l1,feat_l2,feat_l3):
        feat3 = self.decoder3(torch.cat(feat_l3,1))
        up_feat3 = self.up_sampler3(feat3)

        [feat_lt2,feat_rt2,feat_lb2,feat_rb2,mask_lt2,mask_rt2,mask_lb2,mask_rb2] = feat_l2
        input_s2 = torch.cat([feat_lt2,feat_rt2,feat_lb2,feat_rb2,up_feat3,mask_lt2,mask_rt2,mask_lb2,mask_rb2],1)
        feat2 = self.decoder2(input_s2)
        up_feat2 = self.up_sampler2(feat2)

        [feat_lt1,feat_rt1,feat_lb1,feat_rb1,mask_lt1,mask_rt1,mask_lb1,mask_rb1] = feat_l1
        input_s1 = torch.cat([feat_lt1,feat_rt1,feat_lb1,feat_rb1,up_feat2,mask_lt1,mask_rt1,mask_lb1,mask_rb1],1)
        feat1 = self.decoder1(input_s1)

        feat = self.decoder(feat1)
        img_feat = self.conv_img(feat)
        confid_feat = self.lru(self.conv_confid(feat))

        return img_feat,confid_feat

class _Fusion_net(nn.Module):
    def __init__(self):
        super(_Fusion_net, self).__init__()
        self.conv1 = nn.Conv2d(16,8,3,1,1)
        self.conv2 = nn.Conv2d(8,8,3,1,1)
        self.conv3 = nn.Conv2d(8,4,3,1,1)
        self.conv_mask = nn.Conv2d(4,1,3,1,1)
        self.lru = nn.LeakyReLU(0.1,True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, confid_pix,confid_feat,img_pix,img_feat):
        x = torch.cat([confid_pix,confid_feat],1)
        x = self.lru(self.conv1(x))
        x = self.lru(self.conv2(x))
        x = self.lru(self.conv3(x))
        mask = self.sigmoid(self.conv_mask(x))
        output_img = img_pix*mask + img_feat*(1-mask)
        return output_img

class _Feat_encoder(nn.Module):
    def __init__(self):
        super(_Feat_encoder, self).__init__()
        self.vgg_encoder = models.vgg19(pretrained=False).features[:18]
        self.selected_layers = [3,8,17]
        ######### freeze weights ###################
        for param in self.vgg_encoder.parameters():
            param.requires_grad = False
        ############################################
    def forward(self,x):
        feat_list = []
        rgb_mean = [123.6800, 116.7790, 103.9390]
        tensor_mean = torch.tensor(rgb_mean).reshape([1,3,1,1])
        if x.is_cuda:
            tensor_mean = tensor_mean.cuda()
        _x = x * 255. - tensor_mean
        for index, layer in enumerate(self.vgg_encoder):
            _x = layer(_x)
            if (index in self.selected_layers):
                feat_list.append(_x)
        return feat_list

class FPFR(nn.Module):
    def __init__(self):
        super(FPFR, self).__init__()
        self.depth_estimator = PWC_net()
        self.pix_net = _Pix_net()
        self.feat_net = _Feat_net()
        self.fus_net = _Fusion_net()
        self.feat_encoder = _Feat_encoder()
    def depth_estimation(self, lt, rt, lb, rb, D):
        def batch_bwd(img_list,disp,pos_list):
            disp_volume = torch.unsqueeze(disp,1).repeat([1,3,2,1,1]) # [B,3,2,H,W]
            target_img = torch.stack(img_list[0:1],1).repeat([1,3,1,1,1]) # [B,3,C,H,W]
            ref_imgs = torch.stack(img_list[1:],1) # [B,3,C,H,W]
            B,_,C,H,W = ref_imgs.size()
            target_pos = torch.stack(pos_list[0:1],0) # [1,1,2]
            ref_pos = torch.stack(pos_list[1:],0) # [3,1,2]
            relative_displacement = (ref_pos - target_pos).reshape([1,3,1,1,2]).permute([0,1,4,2,3]) # [1,3,2,1,1]
            relative_displacement.float()
            batch_flow_y,batch_flow_x = torch.unbind(disp_volume*relative_displacement,2)
            batch_flow = torch.stack([batch_flow_x,batch_flow_y],2)
            warped_imgs_batch = _bwd2d(ref_imgs.reshape([-1,C,H,W]),batch_flow.reshape([-1,2,H,W]))
            warped_imgs = warped_imgs_batch.reshape([B,3,C,H,W])
            warping_error = torch.sum((warped_imgs-target_img)**2,[1,2],True)
            return warping_error[:,0,...]
        ###################################################################################################
        B,C,H,W = lt.size()
        horizontal_img_volume_left = torch.stack([lt,rt,lb,rb],1).reshape([-1,C,H,W])
        horizontal_img_volume_right = torch.stack([rt,lt,rb,lb],1).reshape([-1,C,H,W])
        vertical_img_volume_left = torch.rot90(horizontal_img_volume_left,1,[2,3])
        vertical_img_volume_right = torch.rot90(torch.stack([lb,rb,lt,rt],1).reshape([-1,C,H,W]),1,[2,3])

        horizontal_flow = self.depth_estimator(horizontal_img_volume_left,horizontal_img_volume_right)[:,0:1,...] # [B*N,1,H,W]
        vertical_flow = torch.rot90(self.depth_estimator(vertical_img_volume_left,vertical_img_volume_right),-1,[2,3])[:,0:1,...] # [B*N,1,H,W]

        disp_lt_1,disp_rt_1,disp_lb_1,disp_rb_1 = torch.unbind(horizontal_flow.reshape([B,4,1,H,W]),1)
        disp_lt_2,disp_rt_2,disp_lb_2,disp_rb_2 = torch.unbind(vertical_flow.reshape([B,4,1,H,W]),1)
        ###################################################################################################
        ind_lt,ind_rt,ind_lb,ind_rb = torch.tensor([0,0]),torch.tensor([0,1]),torch.tensor([1,0]),torch.tensor([1,1])
        if lt.is_cuda:
            ind_lt, ind_rt, ind_lb, ind_rb = ind_lt.cuda(),ind_rt.cuda(),ind_lb.cuda(),ind_rb.cuda()
        error_lt1 = batch_bwd([lt,rt,lb,rb],disp_lt_1,[ind_lt,ind_rt,ind_lb,ind_rb])
        error_rt1 = batch_bwd([rt,lt,lb,rb],-disp_rt_1,[ind_rt,ind_lt,ind_lb,ind_rb])
        error_lb1 = batch_bwd([lb,lt,rt,rb],disp_lb_1,[ind_lb,ind_lt,ind_rt,ind_rb])
        error_rb1 = batch_bwd([rb,rt,lt,lb],-disp_rb_1,[ind_rb,ind_rt,ind_lt,ind_lb])

        error_lt2 = batch_bwd([lt,rt,lb,rb],disp_lt_2,[ind_lt,ind_rt,ind_lb,ind_rb])
        error_rt2 = batch_bwd([rt,lt,lb,rb],disp_rt_2,[ind_rt,ind_lt,ind_lb,ind_rb])
        error_lb2 = batch_bwd([lb,lt,rt,rb],-disp_lb_2,[ind_lb,ind_lt,ind_rt,ind_rb])
        error_rb2 = batch_bwd([rb,rt,lt,lb],-disp_rb_2,[ind_rb,ind_rt,ind_lt,ind_lb])

        mask_lt = (error_lt1 < error_lt2).float()
        disp_lt = (disp_lt_1 / D) * mask_lt + (disp_lt_2 / D) * (torch.ones_like(mask_lt) - mask_lt)

        mask_rt = (error_rt1 < error_rt2).float()
        disp_rt = (-disp_rt_1 / D) * mask_rt + (disp_rt_2 / D) * (torch.ones_like(mask_rt) - mask_rt)

        mask_lb = (error_lb1 < error_lb2).float()
        disp_lb = (disp_lb_1 / D) * mask_lb + (-disp_lb_2 / D) * (torch.ones_like(mask_lb) - mask_lb)

        mask_rb = (error_rb1 < error_rb2).float()
        disp_rb = (-disp_rb_1 / D) * mask_rb + (-disp_rb_2 / D) * (torch.ones_like(mask_rb) - mask_rb)

        return disp_lt,disp_rt,disp_lb,disp_rb

    def group_warping(self,disp_list,feat_list,scale,R1,C1,D):
        [disp_lt,disp_rt,disp_lb,disp_rb] = disp_list
        [feat_lt,feat_rt,feat_lb,feat_rb] = feat_list
        C2 = D - C1
        R2 = D - R1
        resize_disp_lt = F.interpolate(disp_lt*scale, scale_factor=scale, mode='bilinear', align_corners=True)
        resize_disp_rt = F.interpolate(disp_rt*scale, scale_factor=scale, mode='bilinear', align_corners=True)
        resize_disp_lb = F.interpolate(disp_lb*scale, scale_factor=scale, mode='bilinear', align_corners=True)
        resize_disp_rb = F.interpolate(disp_rb*scale, scale_factor=scale, mode='bilinear', align_corners=True)
        flow_lt = torch.cat([resize_disp_lt*C1,resize_disp_lt*R1],1)
        flow_rt = torch.cat([-resize_disp_rt*C2,resize_disp_rt*R1],1)
        flow_lb = torch.cat([resize_disp_lb*C1,-resize_disp_lb*R2],1)
        flow_rb = torch.cat([-resize_disp_rb*C2,-resize_disp_rb*R2],1)
        [color_lt,mask_lt] = _fwd2d(feat_lt,flow_lt,resize_disp_lt,10)
        [color_rt,mask_rt] = _fwd2d(feat_rt,flow_rt,resize_disp_rt,10)
        [color_lb,mask_lb] = _fwd2d(feat_lb,flow_lb,resize_disp_lb,10)
        [color_rb,mask_rb] = _fwd2d(feat_rb,flow_rb,resize_disp_rb,10)

        return [color_lt,color_rt,color_lb,color_rb,1.-mask_lt,1.-mask_rt,1.-mask_lb,1.-mask_rb]


    def forward(self, img_lt,img_rt,img_lb,img_rb,R,C,D):
        # img_lt, img_rt, img_lb, img_rb = X['img_lt'], X['img_rt'], X['img_lb'], X['img_rb']
        R1 = torch.reshape(R.clone().detach(),[-1,1,1,1])
        C1 = torch.reshape(C.clone().detach(),[-1,1,1,1])
        D = torch.reshape(D.clone().detach(),[-1,1,1,1])
        ###################################################################################################################
        [feat1_lt,feat2_lt,feat3_lt] = self.feat_encoder(img_lt)
        [feat1_rt,feat2_rt,feat3_rt] = self.feat_encoder(img_rt)
        [feat1_lb,feat2_lb,feat3_lb] = self.feat_encoder(img_lb)
        [feat1_rb,feat2_rb,feat3_rb] = self.feat_encoder(img_rb)
        disp_list = self.depth_estimation(img_lt,img_rt,img_lb,img_rb, D)
        img_list = [img_lt,img_rt,img_lb,img_rb]
        feat1_list = [feat1_lt,feat1_rt,feat1_lb,feat1_rb]
        feat2_list = [feat2_lt,feat2_rt,feat2_lb,feat2_rb]
        feat3_list = [feat3_lt,feat3_rt,feat3_lb,feat3_rb]
        ###################################################################################################################
        I_list = self.group_warping(disp_list, img_list, 1, R1, C1, D)
        F1_list = self.group_warping(disp_list, feat1_list, 1, R1, C1, D)
        F2_list = self.group_warping(disp_list, feat2_list, 0.5, R1, C1, D)
        F3_list = self.group_warping(disp_list, feat3_list, 0.25, R1, C1, D)
        ###################################################################################################################
        img_pix,confid_pix = self.pix_net(torch.cat(I_list,1))
        img_feat,confid_feat = self.feat_net(F1_list,F2_list,F3_list)
        img_final = self.fus_net(confid_pix,confid_feat,img_pix,img_feat)

        return img_final
