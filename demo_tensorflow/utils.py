#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def cal_padding(img):
    height, width, _ = np.shape(img)
    height_pad = 768
    width_pad = 768
    height_st = np.int32(np.floor(height_pad - height) / 2.)
    width_st = np.int32(np.floor(width_pad - width) / 2.)
    return [height_pad,width_pad,height_st,width_st]
def img_padding(img):
    img = img[...,:3]
    height,width,_ = np.shape(img)
    [height_pad, width_pad, height_st, width_st] = cal_padding(img)
    buffer = np.zeros([height_pad,width_pad,3],np.float32)
    buffer[height_st:(height_st+height),width_st:(width_st+width),:] = img
    return buffer
def img_depadding(img,height_st, width_st,height,width):
    return img[height_st:height_st+height,width_st:width_st+width]
def rot_flip(lt,rt,lb,rb,dlm,h,v):
    rot_0 = np.transpose([[lt,rt],[lb,rb]],[2,3,4,0,1])
    [h_0,v_0] = [h,v]
    rot_1 = np.rot90(np.rot90(rot_0,1,(3,4)),1,(0,1))
    [h_1,v_1] = [v,dlm-h]
    rot_2 = np.rot90(np.rot90(rot_0,2,(3,4)),2,(0,1))
    [h_2,v_2] = [dlm-h,dlm-v]
    rot_3 = np.rot90(np.rot90(rot_0,3,(3,4)),3,(0,1))
    [h_3,v_3] = [dlm-v,h]
    flip_0 = np.flip(rot_0,[1,-1])
    [h_f0,v_f0] = [dlm-h_0,v_0]
    flip_1 = np.flip(rot_1,[1,-1])
    [h_f1,v_f1] = [dlm-h_1,v_1]
    flip_2 = np.flip(rot_2,[1,-1])
    [h_f2,v_f2] = [dlm-h_2,v_2]
    flip_3 = np.flip(rot_3,[1,-1])
    [h_f3,v_f3] = [dlm-h_3,v_3]
    # img_vl = np.stack([rot_0,rot_1,rot_2,rot_3,flip_0,flip_1,flip_2,flip_3],0)
    img_list = [rot_0,rot_1,rot_2,rot_3,flip_0,flip_1,flip_2,flip_3]
    h_vl = np.stack([[h_0],[h_1],[h_2],[h_3],[h_f0],[h_f1],[h_f2],[h_f3]],0)
    v_vl = np.stack([[v_0],[v_1],[v_2],[v_3],[v_f0],[v_f1],[v_f2],[v_f3]],0)
    dlm_vl = np.repeat(np.stack([[dlm]],0),8,0)
    # return [np.float32(img_vl),np.float32(h_vl),np.float32(v_vl),np.float32(dlm_vl)]
    return [img_list,np.float32(h_vl),np.float32(v_vl),np.float32(dlm_vl)]
def derot_flip(img_list):
    img0 = img_list[0]
    img1 = np.rot90(img_list[1],-1)
    img2 = np.rot90(img_list[2],-2)
    img3 = np.rot90(img_list[3],-3)

    img4 = np.flip(img_list[4],1)
    img5 = np.rot90(np.flip(img_list[5],1),-1)
    img6 = np.rot90(np.flip(img_list[6],1),-2)
    img7 = np.rot90(np.flip(img_list[7],1),-3)
    return np.stack([img0,img1,img2,img3,img4,img5,img6,img7],0)
def load_graph(fz_gh_fn):
    with tf.gfile.GFile(fz_gh_fn, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="FPFR"
            )
    return graph