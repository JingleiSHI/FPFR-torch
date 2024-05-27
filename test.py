import torch
import numpy as np
from models.FPFR import FPFR
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# training parameters #
model = FPFR()
model = model.cuda()
model = model.eval()

data_folder = './data/boxes'
result_folder = './results'
model_path = 'saved_model/synth.pkl'
model.load_state_dict(torch.load(model_path))
lt = torch.from_numpy(np.transpose(plt.imread(data_folder+'/' + 'lf_1_1.png')[np.newaxis,...,:3],[0,3,1,2]))
rt = torch.from_numpy(np.transpose(plt.imread(data_folder+'/' + 'lf_1_9.png')[np.newaxis,...,:3],[0,3,1,2]))
lb = torch.from_numpy(np.transpose(plt.imread(data_folder+'/' + 'lf_9_1.png')[np.newaxis,...,:3],[0,3,1,2]))
rb = torch.from_numpy(np.transpose(plt.imread(data_folder+'/' + 'lf_9_9.png')[np.newaxis,...,:3],[0,3,1,2]))
D = 8

with torch.no_grad():
    for R in range(9):
        for C in range(9):
            print(R,C)
            lt, rt, lb, rb, R_index, C_index, D = lt.cuda(), rt.cuda(), lb.cuda(), rb.cuda(), torch.tensor(R).cuda(), torch.tensor(C).cuda(), torch.tensor(D).cuda()
            output = model(lt, rt, lb, rb, R_index, C_index, D)
            image = output[0,...]
            image = torch.permute(image,(1,2,0))
            image = image.cpu().detach().numpy()
            image[image<0] = 0.
            image[image>1] = 1.
            plt.imsave(result_folder+'/lf_'+str(R+1)+'_'+str(C+1)+'.png',image)

