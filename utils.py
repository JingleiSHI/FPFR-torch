#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jinglei Shi
@file: warping_tf.py
@time: 09/04/2020 14:33
'''
import sys
import os
import torch
import torch.nn.functional as F
# from warping_cuda import FunctionSoftsplat

def _fwd2d(source, flow, depth, alpha):
    # source [B,C,H,W]
    # flow [B,2,H,W]
    # depth [B,1,H,W]
    B, C, H, W = source.size()
    # mesh grid
    x = torch.arange(W)
    y = torch.arange(H)
    z = torch.arange(B)
    zz, yy, xx = torch.meshgrid([z, y, x])  # [B,H,W]
    zz = zz.float()
    xx = xx.float()
    yy = yy.float()
    if source.is_cuda:
        xx = xx.cuda()
        yy = yy.cuda()
        zz = zz.cuda()
    x_lim = W - 1.
    y_lim = H - 1.
    fx, fy = torch.unbind(flow, 1)  # [B,H,W]

    _x = xx + fx
    _y = yy + fy

    x0 = torch.floor(_x)
    x1 = x0 + 1
    y0 = torch.floor(_y)
    y1 = y0 + 1

    x0_safe = torch.clamp(x0, 0, x_lim)
    y0_safe = torch.clamp(y0, 0, y_lim)
    x1_safe = torch.clamp(x1, 0, x_lim)
    y1_safe = torch.clamp(y1, 0, y_lim)

    # Bilinear splat weights, with points outside the grid having weight 0.
    wt_x0 = (x1 - _x) * torch.eq(x0, x0_safe).float()
    wt_x1 = (_x - x0) * torch.eq(x1, x1_safe).float()
    wt_y0 = (y1 - _y) * torch.eq(y0, y0_safe).float()
    wt_y1 = (_y - y0) * torch.eq(y1, y1_safe).float()

    wt_tl = torch.unsqueeze(wt_x0 * wt_y0, 1)  # [B,1,H,W]
    wt_tr = torch.unsqueeze(wt_x1 * wt_y0, 1)
    wt_bl = torch.unsqueeze(wt_x0 * wt_y1, 1)
    wt_br = torch.unsqueeze(wt_x1 * wt_y1, 1)

    # Clamp small weights to zero for gradient numerical stability (IMPORTANT!)
    wt_tl *= torch.gt(wt_tl, 1e-3).float()
    wt_tr *= torch.gt(wt_tr, 1e-3).float()
    wt_bl *= torch.gt(wt_bl, 1e-3).float()
    wt_br *= torch.gt(wt_br, 1e-3).float()

    min_vector, _ = torch.min(torch.reshape(depth, [B, -1]), 1)
    min_value = torch.reshape(min_vector, [B, 1, 1, 1]).repeat(1, 1, H, W)
    max_vector, _ = torch.max(torch.reshape(depth, [B, -1]), 1)
    max_value = torch.reshape(max_vector, [B, 1, 1, 1]).repeat(1, 1, H, W)
    normalized_disparity = torch.div(depth - min_value, max_value - min_value)
    weights = torch.exp(- alpha * torch.abs(normalized_disparity))
    img_tl = source * weights * wt_tl
    img_tr = source * weights * wt_tr
    img_bl = source * weights * wt_bl
    img_br = source * weights * wt_br

    w_tl = (weights * wt_tl).permute([0, 2, 3, 1])
    w_tr = (weights * wt_tr).permute([0, 2, 3, 1])
    w_bl = (weights * wt_bl).permute([0, 2, 3, 1])
    w_br = (weights * wt_br).permute([0, 2, 3, 1])

    indices_tl = torch.stack([zz, y0_safe, x0_safe], -1).long()
    indices_tr = torch.stack([zz, y0_safe, x1_safe], -1).long()
    indices_bl = torch.stack([zz, y1_safe, x0_safe], -1).long()
    indices_br = torch.stack([zz, y1_safe, x1_safe], -1).long()

    img_tl = img_tl.permute([0, 2, 3, 1])
    img_tr = img_tr.permute([0, 2, 3, 1])
    img_bl = img_bl.permute([0, 2, 3, 1])
    img_br = img_br.permute([0, 2, 3, 1])

    buffer_tl, buffer_tr, buffer_bl, buffer_br = torch.zeros_like(img_tl), torch.zeros_like(img_tr), torch.zeros_like(img_bl), torch.zeros_like(img_br)
    weights_tl, weights_tr, weights_bl, weights_br = torch.zeros_like(w_tl), torch.zeros_like(w_tr), torch.zeros_like(w_bl), torch.zeros_like(w_br)
    if source.is_cuda:
        buffer_tl, buffer_tr, buffer_bl, buffer_br = buffer_tl.cuda(), buffer_tr.cuda(), buffer_bl.cuda(), buffer_br.cuda()
        weights_tl, weights_tr, weights_bl, weights_br = weights_tl.cuda(), weights_tr.cuda(), weights_bl.cuda(), weights_tr.cuda()
    warped_color = buffer_tl.index_put(tuple(indices_tl.reshape(-1, 3).t()), img_tl.reshape(-1, C), True) + \
                   buffer_tr.index_put(tuple(indices_tr.reshape(-1, 3).t()), img_tr.reshape(-1, C), True) + \
                   buffer_bl.index_put(tuple(indices_bl.reshape(-1, 3).t()), img_bl.reshape(-1, C), True) + \
                   buffer_br.index_put(tuple(indices_br.reshape(-1, 3).t()), img_br.reshape(-1, C), True)

    warped_weights = weights_tl.index_put(tuple(indices_tl.reshape(-1, 3).t()), w_tl.reshape(-1, 1), True) + \
                     weights_tr.index_put(tuple(indices_tr.reshape(-1, 3).t()), w_tr.reshape(-1, 1), True) + \
                     weights_bl.index_put(tuple(indices_bl.reshape(-1, 3).t()), w_bl.reshape(-1, 1), True) + \
                     weights_br.index_put(tuple(indices_br.reshape(-1, 3).t()), w_br.reshape(-1, 1), True)
    ########### For safe division ######################################
    mask = torch.gt(warped_weights, 0.).float().permute([0, 3, 1, 2])
    warped_weights = warped_weights + 1e-8 * torch.eq(warped_weights, torch.tensor(0.)).float()
    warped_img = torch.div(warped_color, warped_weights).permute([0, 3, 1, 2])

    return warped_img, mask

def _fwd2d_cuda(source, flow, depth, alpha):
    nor_depth = (depth - torch.min(depth))/(torch.max(depth) - torch.min(depth))
    warped_img = FunctionSoftsplat(source,flow,-alpha*nor_depth)
    buffer = torch.ones_like(source).cuda()
    mask = FunctionSoftsplat(buffer,flow,-alpha*nor_depth)
    return warped_img, mask[:,0:1,...]

def _bwd2d(source, flow):
    # source [B,C,H,W]
    # flow [B,2,H,W]
    B, C, H, W = source.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if source.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(source, vgrid, padding_mode='border')    
    mask = torch.ones(source.size())
    if source.is_cuda:
        mask = mask.cuda()
    mask = F.grid_sample(mask, vgrid, padding_mode='border')

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask

