"""
here we define a customized dataset that can be used for torch dataloader to read data,
! All preprocessing for data should be done here !
"""
from torch.utils.data import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt

def trainset_loader(path, H, W, Dim, Min_len):
    D = np.random.randint(Min_len-1, Dim)
    row, col = np.random.randint(1, Dim - D + 1, 2)
    R, C = np.random.randint(0, D + 1, 2)
    lt = plt.imread(path+'/' + 'lf_' + str(row) + '_' + str(col) + '.png')[...,:3]
    height,width,_ = np.shape(lt)
    assert H <= height and W <= width, 'The size of path is too large.'
    H0 = np.random.randint(0, height - H + 1)
    W0 = np.random.randint(0, width - W + 1)
    lt = np.transpose(lt[H0:(H0+H),W0:(W0+W),:],[2,0,1])
    rt = np.transpose(plt.imread(path+'/' + 'lf_' + str(row) + '_' + str(col+D) + '.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    lb = np.transpose(plt.imread(path+'/' + 'lf_' + str(row+D) + '_' + str(col) + '.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    rb = np.transpose(plt.imread(path+'/' + 'lf_' + str(row+D) + '_' + str(col+D) + '.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    tgt = np.transpose(plt.imread(path+'/' + 'lf_' + str(row+R) + '_' + str(col+C) + '.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])

    return lt, rt, lb, rb, tgt, R, C, D

def testset_loader(path, H, W, Dim, Min_len):
    lt = plt.imread(path+'/' + 'lf_2_2.png')[...,:3]
    height,width,_ = np.shape(lt)
    assert H <= height and W <= width, 'The size of path is too large.'
    H0 = np.int(np.floor((height - H)/2.))
    W0 = np.int(np.floor((width - W)/2.))
    lt = np.transpose(lt[H0:(H0+H),W0:(W0+W),:],[2,0,1])
    rt = np.transpose(plt.imread(path+'/' + 'lf_2_8.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    lb = np.transpose(plt.imread(path+'/' + 'lf_8_2.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    rb = np.transpose(plt.imread(path+'/' + 'lf_8_8.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    tgt = np.transpose(plt.imread(path+'/' + 'lf_5_5.png')[H0:(H0+H),W0:(W0+W),:3],[2,0,1])
    R, C, D = 3, 3, 6
    return lt, rt, lb, rb, tgt, R, C, D

class GeneralDataset(Dataset):
    def __init__(self, data_file_path, folder_path, sample_ratio = 1, patch_size = [100,100], dimension = 9, min_len = 7, loader = trainset_loader):
        super(GeneralDataset).__init__()
        assert sample_ratio <= 1. and sample_ratio > 0., 'Sample ration should be between 0 and 1.'
        assert min_len <= dimension and min_len > 2, 'Min dim should be inferior to Dim'
        self.file = open(data_file_path)
        self.folder_path = folder_path
        self.raw_list = self.file.read().splitlines()
        number = np.int(len(self.raw_list) * sample_ratio)
        random.shuffle(self.raw_list)
        self.data_list = self.raw_list[:number]
        [self.H, self.W] = patch_size
        self.dimension = dimension
        self.min_len = min_len
        self.loader = loader

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        scene_name = self.data_list[index]
        scene_path = self.folder_path + '/' + scene_name
        lt, rt, lb, rb, tgt, R, C, D = self.loader(scene_path, self.H, self.W, self.dimension, self.min_len)
        return lt, rt, lb, rb, tgt, R, C, D
