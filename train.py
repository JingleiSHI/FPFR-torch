import torch
import numpy as np
import pickle
import torch.optim as optim
from losses.Loss_LF import Loss_L1
from models.FPFR import FPFR
from torch.utils.data import DataLoader
from metrics.metrics_torch import PSNR,SSIM
from datasets.Dataset_LF import GeneralDataset,trainset_loader,testset_loader
# training parameters #
#-------------------------------------------------
lr = 0.000001
train_list = 'datasets/tri_trainlist.txt'
val_list = 'datasets/tri_testlist.txt'
train_batchsize = 4
val_batchsize = 1
train_patchsize = [160,160]
val_patchsize = [768,768]
train_ratio = 1.
val_ratio = 1.
Dim = 9
Min_len = 7
#-------------------------------------------------
num_workers = train_batchsize*4
use_gpu = True
firstTrain = False
test_frequency = 1
save_frequency = 100
trainset_folder = '/xxx'
testset_folder = '/xxx'
#-------------------------------------------------
version_dir = 'saved_model/test'
best_dir = version_dir + '/best.pkl'
regular_dir = version_dir + '/regular.pkl'
#-------------------------------------------------

# create dataloader to read data from YourData #
train_dataset = GeneralDataset(train_list,trainset_folder,train_ratio,train_patchsize,Dim,Min_len,trainset_loader)
val_dataset = GeneralDataset(val_list,testset_folder,val_ratio,val_patchsize,Dim,Min_len,testset_loader)
train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_workers, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=val_batchsize, shuffle=True, num_workers=num_workers, drop_last=False)

# init your models #
model = FPFR()
model = model.train()
if use_gpu:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
myloss = Loss_L1()
cal_psnr = PSNR()
cal_ssim = SSIM()

if firstTrain:
    print('Initialization of the pipeline.')
    model.load_state_dict(torch.load('saved_model/tf/synth.pkl'))
    best_psnr = 0.
    epoch = 0
    learning_curve = []
else:
    print('Loading the saved model.')
    checkpoint = torch.load(best_dir)
    model.load_state_dict(checkpoint['net'])
    epoch = checkpoint['epoch'] + 1
    best_psnr = checkpoint['best_psnr']
    learning_curve = pickle.load(open(version_dir + '/learning_curve.dat','rb'))

# start to train #
print('Starting training.')
while True:
    for iter_i, [lt,rt,lb,rb,tgt,R,C,D] in enumerate(train_dataloader):
        if use_gpu:
            lt, rt, lb, rb, tgt, R, C, D = lt.cuda(), rt.cuda(), lb.cuda(), rb.cuda(), tgt.cuda(), R.cuda(), C.cuda(), D.cuda()
        output = model(lt, rt, lb, rb, R, C, D)
        loss = myloss(output, tgt)
        # before doing backprop clean up gradients in variables from previous iteration
        model.zero_grad()
        # back propagation of this loss
        loss.backward()
        # update your parameters by gradient descent
        optimizer.step()
    epoch += 1
    if epoch % test_frequency == 0:
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for iter_j, [lt_v,rt_v,lb_v,rb_v,tgt_v,R_v,C_v,D_v] in enumerate(val_dataloader):
                if use_gpu:
                    lt_v, rt_v, lb_v, rb_v, tgt_v, R_v, C_v, D_v = lt_v.cuda(),rt_v.cuda(),lb_v.cuda(),rb_v.cuda(),tgt_v.cuda(),R_v.cuda(),C_v.cuda(),D_v.cuda()
                output = model(lt_v,rt_v,lb_v,rb_v,R_v,C_v,D_v)
                psnr,ssim = cal_psnr(output,tgt_v), cal_ssim(output,tgt_v)
                psnr_list = psnr_list + list(psnr.cpu().numpy())
                ssim_list = ssim_list + list(ssim.cpu().numpy())
            mean_psnr, mean_ssim = np.mean(psnr_list), np.mean(ssim_list)
            print('Epoch ', epoch, 'PSNR ',mean_psnr,' SSIM ',mean_ssim)
            learning_curve.append([mean_psnr,mean_ssim])
            if mean_psnr > best_psnr:
                state = {'net': model.state_dict(), 'best_psnr': best_psnr, 'epoch':epoch}
                torch.save(state, best_dir)
                f1 = open(version_dir +'/learning_curve.dat', 'wb')
                pickle.dump(learning_curve, f1)
                f1.close()
        if epoch % save_frequency == 0:
            state = {'net': model.state_dict(), 'best_psnr': best_psnr, 'epoch': epoch}
            torch.save(state, regular_dir)
