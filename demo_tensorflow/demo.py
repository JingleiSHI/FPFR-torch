#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from utils import *
import argparse
import os

def run_demo(scene_name,mode='FPFR*',data_type='lytro',angular_resolution=7,inter_extra='inter'):
    if inter_extra == 'inter':
        if data_type == 'synthetic':
            print('Synthetic Light Field')
            graph = load_graph('demo_synth.pb')
            [row,column,d] = [2,2,6]
        else:
            print('Lytro Light Field')
            graph = load_graph('demo_lytro.pb')
            [row,column,d] = [1,1,7]

        assert type(angular_resolution) == int and angular_resolution >= 3, 'Angular resolution should be an integer greater or equal to 3'

        img_lt = graph.get_tensor_by_name('FPFR/Placeholder:0')
        img_rt = graph.get_tensor_by_name('FPFR/Placeholder_1:0')
        img_lb = graph.get_tensor_by_name('FPFR/Placeholder_2:0')
        img_rb = graph.get_tensor_by_name('FPFR/Placeholder_3:0')
        displacement = graph.get_tensor_by_name('FPFR/Placeholder_4:0')
        horizontal = graph.get_tensor_by_name('FPFR/Placeholder_5:0')
        vertical = graph.get_tensor_by_name('FPFR/Placeholder_6:0')
        view_synth = graph.get_tensor_by_name('FPFR/Confidence_fusion/add:0')
        sess = tf.Session(graph=graph)
        print('Reading demo scene '+scene_name)

        im = plt.imread('scenes/interpolation/'+scene_name+'/lf_'+str(row)+'_'+str(column)+'.png')[...,:3]
        height,width,_ = np.shape(im)
        [H,W,H_st,W_st] = cal_padding(im)
        lt = img_padding(plt.imread('scenes/interpolation/'+scene_name+'/lf_'+str(row)+'_'+str(column)+'.png'))
        rt = img_padding(plt.imread('scenes/interpolation/'+scene_name+'/lf_'+str(row)+'_'+str(column+d)+'.png'))
        lb = img_padding(plt.imread('scenes/interpolation/'+scene_name+'/lf_'+str(row+d)+'_'+str(column)+'.png'))
        rb = img_padding(plt.imread('scenes/interpolation/'+scene_name+'/lf_'+str(row+d)+'_'+str(column+d)+'.png'))

        if mode == 'FPFR*':
            print('Running FPFR*, generating light field of size '+str(angular_resolution)+' x '+str(angular_resolution))
            for row_counter in range(angular_resolution):
                for column_counter in range(angular_resolution):
                    h = column_counter*(d/(angular_resolution-1.))
                    v = row_counter*(d/(angular_resolution-1.))
                    [img_vl,h_vl,v_vl,dlm_vl] = rot_flip(lt,rt,lb,rb,d,h,v)
                    synthesis_list = []
                    for i in range(np.shape(h_vl)[0]):
                        img_h,img_w,_,_,_ = np.shape(img_vl[i])
                        final_view = sess.run(view_synth, feed_dict={img_lt: img_vl[i][np.newaxis,...,0,0],
                                                                     img_rt: img_vl[i][np.newaxis,...,0,1],
                                                                     img_lb: img_vl[i][np.newaxis,...,1,0],
                                                                     img_rb: img_vl[i][np.newaxis,...,1,1],
                                                                     displacement:dlm_vl[i:(i+1),...],
                                                                     horizontal:h_vl[i:(i+1),...],
                                                                     vertical:v_vl[i:(i+1),...]})
                        synthesis_list.append(final_view[0, ...])
                    img_vl_decod = derot_flip(synthesis_list)
                    img_mix = img_depadding(np.mean(img_vl_decod,0),H_st,W_st,height,width)
                    img_mix[img_mix < 0.] = 0.
                    img_mix[img_mix > 1.] = 1.
                    folder_path = './results/'+scene_name+'/'
                    folder = os.path.exists(folder_path)
                    if not folder:
                        os.makedirs(folder_path)
                    plt.imsave(folder_path+'/'+scene_name+'_inter_'+str(row_counter+1)+'_'+str(column_counter+1)+'_FPFR_star.png',img_mix)
            print('Prediction accomplished.')
        else:
            print('Running FPFR, generating light field of size '+str(angular_resolution)+' x '+str(angular_resolution))
            for row_counter in range(angular_resolution):
                for column_counter in range(angular_resolution):
                    h = column_counter*(d/(angular_resolution-1.))
                    v = row_counter*(d/(angular_resolution-1.))
                    final_view = sess.run(view_synth, feed_dict={img_lt: lt[np.newaxis,...],
                                                                 img_rt: rt[np.newaxis,...],
                                                                 img_lb: lb[np.newaxis,...],
                                                                 img_rb: rb[np.newaxis,...],
                                                                 displacement:np.reshape(d,[1,1]),
                                                                 horizontal: np.reshape(h,[1,1]),
                                                                 vertical: np.reshape(v,[1,1])})
                    img_sig = img_depadding(final_view[0,...],H_st,W_st,height,width)
                    img_sig[img_sig < 0.] = 0.
                    img_sig[img_sig > 1.] = 1.
                    folder_path = './results/' + scene_name + '/'
                    folder = os.path.exists(folder_path)
                    if not folder:
                        os.makedirs(folder_path)
                    plt.imsave(folder_path+'/'+scene_name+'_inter_'+str(row_counter+1)+'_'+str(column_counter+1)+'_FPFR.png',img_sig)
            print('Prediction accomplished.')
    else:
        print('Synthetic Light Field')
        graph = load_graph('demo_synth.pb')
        [row,column,d] = [4,4,2]

        assert type(angular_resolution) == int and angular_resolution%2 == 1 and angular_resolution>3, 'Angular resolution should be an odd integer and greater than 3'

        img_lt = graph.get_tensor_by_name('FPFR/Placeholder:0')
        img_rt = graph.get_tensor_by_name('FPFR/Placeholder_1:0')
        img_lb = graph.get_tensor_by_name('FPFR/Placeholder_2:0')
        img_rb = graph.get_tensor_by_name('FPFR/Placeholder_3:0')
        displacement = graph.get_tensor_by_name('FPFR/Placeholder_4:0')
        horizontal = graph.get_tensor_by_name('FPFR/Placeholder_5:0')
        vertical = graph.get_tensor_by_name('FPFR/Placeholder_6:0')
        view_synth = graph.get_tensor_by_name('FPFR/Confidence_fusion/add:0')
        sess = tf.Session(graph=graph)
        print('Reading demo scene '+scene_name)

        im = plt.imread('scenes/extrapolation/'+scene_name+'/lf_'+str(row)+'_'+str(column)+'.png')[...,:3]
        height,width,_ = np.shape(im)
        [H,W,H_st,W_st] = cal_padding(im)
        lt = img_padding(plt.imread('scenes/extrapolation/'+scene_name+'/lf_'+str(row)+'_'+str(column)+'.png'))
        rt = img_padding(plt.imread('scenes/extrapolation/'+scene_name+'/lf_'+str(row)+'_'+str(column+d)+'.png'))
        lb = img_padding(plt.imread('scenes/extrapolation/'+scene_name+'/lf_'+str(row+d)+'_'+str(column)+'.png'))
        rb = img_padding(plt.imread('scenes/extrapolation/'+scene_name+'/lf_'+str(row+d)+'_'+str(column+d)+'.png'))

        if mode == 'FPFR*':
            print('Running FPFR*, extrapolating light field of size '+str(angular_resolution)+' x '+str(angular_resolution))
            D = (d/2.) - np.floor(angular_resolution/2.)
            for row_counter in range(angular_resolution):
                for column_counter in range(angular_resolution):
                    h = column_counter + D
                    v = row_counter + D
                    [img_vl,h_vl,v_vl,dlm_vl] = rot_flip(lt,rt,lb,rb,d,h,v)
                    synthesis_list = []
                    for i in range(np.shape(h_vl)[0]):
                        img_h,img_w,_,_,_ = np.shape(img_vl[i])
                        final_view = sess.run(view_synth, feed_dict={img_lt: img_vl[i][np.newaxis,...,0,0],
                                                                     img_rt: img_vl[i][np.newaxis,...,0,1],
                                                                     img_lb: img_vl[i][np.newaxis,...,1,0],
                                                                     img_rb: img_vl[i][np.newaxis,...,1,1],
                                                                     displacement:dlm_vl[i:(i+1),...],
                                                                     horizontal:h_vl[i:(i+1),...],
                                                                     vertical:v_vl[i:(i+1),...]})
                        synthesis_list.append(final_view[0, ...])
                    img_vl_decod = derot_flip(synthesis_list)
                    img_mix = img_depadding(np.mean(img_vl_decod,0),H_st,W_st,height,width)
                    img_mix[img_mix < 0.] = 0.
                    img_mix[img_mix > 1.] = 1.
                    folder_path = './results/'+scene_name+'/'
                    folder = os.path.exists(folder_path)
                    if not folder:
                        os.makedirs(folder_path)
                    plt.imsave(folder_path+'/'+scene_name+'_extra_'+str(row_counter+1)+'_'+str(column_counter+1)+'_FPFR_star.png',img_mix)
            print('Prediction accomplished.')
        else:
            print('Running FPFR, extrapolating light field of size '+str(angular_resolution)+' x '+str(angular_resolution))
            D = (d/2.) - np.floor(angular_resolution/2.)
            for row_counter in range(angular_resolution):
                for column_counter in range(angular_resolution):
                    h = column_counter + D
                    v = row_counter + D
                    final_view = sess.run(view_synth, feed_dict={img_lt: lt[np.newaxis,...],
                                                                 img_rt: rt[np.newaxis,...],
                                                                 img_lb: lb[np.newaxis,...],
                                                                 img_rb: rb[np.newaxis,...],
                                                                 displacement:np.reshape(d,[1,1]),
                                                                 horizontal: np.reshape(h,[1,1]),
                                                                 vertical: np.reshape(v,[1,1])})
                    img_sig = img_depadding(final_view[0,...],H_st,W_st,height,width)
                    img_sig[img_sig < 0.] = 0.
                    img_sig[img_sig > 1.] = 1.
                    folder_path = './results/' + scene_name + '/'
                    folder = os.path.exists(folder_path)
                    if not folder:
                        os.makedirs(folder_path)
                    plt.imsave(folder_path+'/'+scene_name+'_extra_'+str(row_counter+1)+'_'+str(column_counter+1)+'_FPFR.png',img_sig)
            print('Prediction accomplished.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_name",
        type=str,
        default="stilllife",
        help="The name of light field scene")
    parser.add_argument(
        "--mode",
        type=str,
        default="FPFR*",
        help="Prediction mode: FPFR or FPFR*")
    parser.add_argument(
        "--data_type",
        type=str,
        default="synthetic",
        help="light field data type: synthetic or lytro")
    parser.add_argument(
        "--angular_resolution",
        type=int,
        default=7,
        help="Angular resolution of generated light field")
    parser.add_argument(
        "--inter_extra",
        type=str,
        default='inter',
        help="Interpolation or extrapolation")

    FLAGS, unparsed = parser.parse_known_args()

    run_demo(FLAGS.scene_name,
             FLAGS.mode,
             FLAGS.data_type,
             FLAGS.angular_resolution,
             FLAGS.inter_extra)
