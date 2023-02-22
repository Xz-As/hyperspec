from __future__ import print_function
from __future__ import division

import torch
import torch.utils.data as data
import torchvision
from lib.network_hyper import Network,Network_1Dconv
from lib.network_hyper_SLIC_nonlocal import Network_div,SLIC
from math import log
from torch import nn
import torch.nn.functional as F
import time
import scipy.io as scio

import heapq

# load data
# ###########################################################################################
# Python 2/3 compatiblity

from torchsummary import summary
# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm       
import sklearn.model_selection
from skimage import io
import scipy.io as sio
# Visualization
import seaborn as sns
import visdom
import random

from Hyperspectral_Classification_master.utils_equal_number import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device
from Hyperspectral_Classification_master.datasets_add_pos import get_dataset, HyperX, open_file, DATASETS_CONFIG
from Hyperspectral_Classification_master.models import get_model, train, test, save_model

from itertools import chain

import argparse
import os

# dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]
dataset_names = 'IndianPines'
# dataset_names = 'PaviaU'
model_names = 'nonlocalnetwork'

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default= dataset_names, help="Dataset to use.")
parser.add_argument('--model', type=str, default= model_names,
                    help="Model to train. Available:\n"
                    "SVM (linear), "
                    "SVM_grid (grid search on linear, poly and RBF kernels), "
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (3D semi-supervised CNN), "
                    "mou (1D RNN),"
                    "nonlocalnetwork")
parser.add_argument('--folder', type=str, help="Folder where to store the datasets (defaults to the current working directory).", default=r"data")
parser.add_argument('--cuda', type=int, default=0, help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None, help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=20, help="Percentage of samples to use for training (default: 10%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode (random sampling or disjoint, default: random)", default='random')
group_dataset.add_argument('--train_set', type=str, default=None, help="Path to the train ground truth (optional, this supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None, help="Path to the test set (optional, by default the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if absent will be set by the model)")
group_train.add_argument('--patch_size', type=int, help="Size of the spatial neighbourhood (optional, if absent will be set by the model)")
group_train.add_argument('--lr', type=float, help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true', help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int, help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1, help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true', help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+', choices=dataset_names, help="Download the specified datasets and quits.")

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

SAMPLE_PERCENTAGE = args.training_sample # % of training samples
FLIP_AUGMENTATION = args.flip_augmentation # Data augmentation ?
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
DATASET = args.dataset # Dataset name
MODEL = args.model # Model name
N_RUNS = args.runs # Number of runs (for cross-validation)
PATCH_SIZE = args.patch_size # Spatial context size (number of neighbours in each spatial direction)
DATAVIZ = args.with_exploration # Add some visualization of the spectra ?
FOLDER = args.folder # Target folder to store/download/load the datasets
EPOCH = args.epoch  # Number of epochs to run
SAMPLING_MODE = args.sampling_mode  # Sampling mode, e.g random sampling
CHECKPOINT = args.restore  # Pre-computed weights to restore
LEARNING_RATE = args.lr  # Learning rate for the SGD
CLASS_BALANCING = args.class_balancing  # Automated class balancing
TRAIN_GT = args.train_set # Training ground truth file
TEST_GT = args.test_set # Testing ground truth file
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

print('DATASET', DATASET)
print('MODEL', MODEL)
viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
# Number of classes
N_CLASSES = len(LABEL_VALUES) -  len(IGNORED_LABELS)
# N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]
print('N_CLASSES:', N_CLASSES)
print('N_BANDS:', N_BANDS)

# Generate super pixel
#n_segments = 1000
#compactness = 1
#super_pixel=SLIC(img , n_segments , compactness)

def Con2Numpy(var_name):
    path = './/slic//'
    dataFile = path + var_name 
    data = scio.loadmat(dataFile)  
    x = data[var_name]
    x1 = x.astype(float)
    return x1

super_pixel = np.array(Con2Numpy('useful_sp_lab'), dtype='int')


if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
#display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
#color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS)
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')

results = []
# run the experiment several times

if TRAIN_GT is not None and TEST_GT is not None:
    train_gt = open_file(TRAIN_GT)
    test_gt = open_file(TEST_GT)
elif TRAIN_GT is not None:
    train_gt = open_file(TRAIN_GT)
    test_gt = np.copy(gt)
    w, h = test_gt.shape
    test_gt[(train_gt > 0)[:w,:h]] = 0
elif TEST_GT is not None:
    test_gt = open_file(TEST_GT)
else:
# Sample random training spectra
    # train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    train_gt, test_gt, gt = sample_gt(gt, SAMPLE_PERCENTAGE, DATASET, mode=SAMPLING_MODE)
#print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
N_CLASSES = np.max(gt)
# print("Running an experiment with the {} model".format(MODEL), "run {}/{}".format(run + 1, N_RUNS))

#display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
#display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")


model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
if CLASS_BALANCING:
    weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
    hyperparams['weights'] = torch.from_numpy(weights)
# Split train set in train/val
# train_gt, val_gt = sample_gt(train_gt, 0.95, mode='random')
# ############################################################################################

# Extract SLIC area
def find_slic(row, col,slic_gt):
    position = np.where(super_pixel == super_pixel[row, col])

    for i in range(0,len(position[0])):
        x = position[0][i]
        y = position[1][i]
        slic_gt[x,y] = gt[x,y]
    return slic_gt
 
# expend train_gt with slic
indices = train_gt.nonzero()
indices = [list(t) for t in zip(*indices)]
indices = np.array(indices)

acc=0
total_len=0
selected=0

for i in indices:
    r,c = i
    position = np.where(super_pixel == super_pixel[r, c])
    total_len+=len(position[0])
    for i in range(0,len(position[0])):
        x = position[0][i]
        y = position[1][i]

        if train_gt[x,y]!=0:
            selected+=1
        
        if(gt[x,y]!=0):
             train_gt[x,y] = gt[r,c]

        if train_gt[x,y] == gt[x,y]:
            acc+=1
         
print('slic acc: %.4f' % (acc*1.0/total_len))
#print('selected: %d' % (selected))

# Generate the dataset
train_dataset = HyperX(img, train_gt, **hyperparams)
train_loader = data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'],
                               # pin_memory=hyperparams['device'],
                               shuffle=True)

test_dataset = HyperX(img, gt, **hyperparams)
test_loader = data.DataLoader(test_dataset, batch_size=hyperparams['batch_size'],
                               #pin_memory=hyperparams['device'],
                               shuffle=False)

gt = torch.from_numpy(gt)

# ############################################################################################
for time_index in range(0, 1):
    i = str(time_index)
    weight_path = os.path.join('weights', i)
    result_path = os.path.join('results', i)
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    #net = Network()
    net = Network_1Dconv()
    net2 = Network_div()

    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    opt2 = torch.optim.Adam(net2.parameters(), lr=0.001)
           
    loss_func = nn.CrossEntropyLoss()
    loss_func1 = nn.LogSoftmax(dim=1)
    loss_low1 = 10
    loss_low2 = 10
    cube_size = 3
    loss1, loss_un, loss2 = 10, 10, 10
    # train stage1 ##################################################################################
    retrain_stage1=False

    if retrain_stage1 == True:
      print('train stage1...')
   
      for epoch_index in range(200):
        st = time.time()

        torch.set_grad_enabled(True)
        net.train()
        net2.train()
        total_loss_train = []
        total_acc_train = 0
        total_sample_train = 0

        for train_batch_index, (img_batch, label_batch, indices) in enumerate(train_loader):

            # add noise
            add_label_noise = False
            if add_label_noise:
                label_batch = label_batch.numpy()
                ratio = 0.1
                # random.seed(0)
                list_lb = random.sample(range(0, label_batch.shape[0]), int(ratio * label_batch.shape[0]))
                label_batch[list_lb] = np.random.randint(1,10, size=int(ratio * label_batch.shape[0]))
                label_batch = torch.from_numpy(label_batch)
            # add noise

            # add a new kind of training set
            img_batch_o = img_batch
            label_batch_o = label_batch
            label_batch_o = label_batch_o - 1 
            indices_o = indices

            net.to(hyperparams['device'])
            net2.to(hyperparams['device'])  

            if epoch_index < 0:
                    # build up slic dataset
                    slic_gt=np.zeros_like(gt.cpu())
                    for i in indices_o:
                            row,col = i
                            slic_gt = find_slic(row,col,slic_gt)
                    slic_dataset = HyperX(img, slic_gt, **hyperparams)
                    slic_loader = data.DataLoader(slic_dataset, batch_size=5000, shuffle=True)             

                    for train_slic_index, (img_batch, label_batch, indices) in enumerate(slic_loader):
                        label_batch = label_batch - 1

                        # if torch.cuda.is_available():
                        img_batch_o = img_batch_o.to(hyperparams['device'])  #cuda()
                        img_batch = img_batch.to(hyperparams['device'])
                        label_batch = label_batch.to(hyperparams['device'])
                        label_batch_o = label_batch_o.to(hyperparams['device'])
                        indices_o = indices_o.to(hyperparams['device'])
                        indices = indices.to(hyperparams['device'])
                        gt = gt.to(hyperparams['device'])

                        predict = net(img_batch_o)  
                        loss1 = loss_func(predict, label_batch_o)

                        predict_un = net(img_batch)  
                        predict_un = F.softmax(predict_un, dim=1)
                        loss_un = loss_func1(predict_un)     # + 1e-8
                        loss_un = -torch.sum(torch.mean(predict_un * loss_un, dim=0, keepdim=True))
                        loss = loss1 + loss_un

                        net.zero_grad()
                        loss.backward()
                        opt.step()

                        predict = predict.argmax(dim=1)
                        acc = (predict == label_batch_o).sum()

                        total_loss_train.append(loss)
                        total_acc_train += acc
                        total_sample_train += img_batch_o.size(0)

            else:   
                    for params in net.parameters():
                        params.requires_grad = False
                                       
                    batch_predict = torch.Tensor()
                    batch_label = torch.LongTensor()

                    indices_o = indices_o.to(hyperparams['device'])
                    batch_predict = batch_predict.to(hyperparams['device'])
                    batch_label = batch_label.to(hyperparams['device'])

                    # build up slic dataset
                    # dataset just contains surrounding area of one pixel
                    for i in indices_o:
                        row,col = i
                        slic_gt = np.zeros_like(gt.cpu())                      
                        slic_gt = find_slic(row,col,slic_gt)
                        slic_dataset = HyperX(img, slic_gt, **hyperparams)
                        slic_loader = data.DataLoader(slic_dataset, batch_size=50000, shuffle=True)             

                        for index, (slic_img_batch, slic_label_batch, slic_indices) in enumerate(slic_loader):
                        
                            slic_img_batch = slic_img_batch.to(hyperparams['device'])
                            slic_label_batch = slic_label_batch.to(hyperparams['device'])
                            slic_indices = slic_indices.to(hyperparams['device'])

                            predict_cnn = net(slic_img_batch)
                            predict , mark = net2(predict_cnn, slic_indices, row, col)
                            predict=predict[mark]
                            slic_label =  slic_label_batch[mark:mark+1]-1

                            batch_predict=torch.cat((batch_predict, predict), 0)
                            batch_label=torch.cat((batch_label, slic_label), 0)

                    loss2 = loss_func(batch_predict, batch_label)

                    net2.zero_grad()
                    loss2.backward()
                    opt2.step()
                   
                    batch_predict = batch_predict.argmax(dim=1)
                    acc = (batch_predict == batch_label).sum()

                    total_loss_train.append(loss2)
                    total_acc_train += acc
                            
                    total_sample_train += img_batch_o.size(0)
                
            # 20200618
            if loss1 < loss_low1:
                weight_path1 = os.path.join(weight_path, 'net.pth')
                # print('Save Net weights to', weight_path1)
                net_save = net.cpu()
                torch.save(net_save.state_dict(), weight_path1)
                loss_low1 = loss1

            if loss2 < loss_low2:
                weight_path1 = os.path.join(weight_path, 'net2.pth')
                net2_save = net2.cpu()
                torch.save(net2_save.state_dict(), weight_path1)
                loss_low2 = loss2

        if epoch_index <= 1000:
            mean_acc_train = total_acc_train.item() * 1.0 / total_sample_train
            mean_loss_train = sum(total_loss_train) / total_loss_train.__len__()

            print('[Train] epoch[%d/%d] acc:%.4f%% loss:%.4f loss1:%.4f loss_un:%.4f loss2:%.4f'
              % (epoch_index, 10, mean_acc_train * 100, mean_loss_train.item(), loss1, loss_un, loss2))

            print('(LR:%f) Time of a epoch:%.4fs\n' % (opt.param_groups[0]['lr'], time.time() - st))

        torch.set_grad_enabled(False)

        net.train()
        net2.train()

    else:
        # for transductive
        # load net parameters
        weight_path2 = os.path.join(weight_path, 'net.pth')
        weight_path2_div = os.path.join(weight_path, 'net2.pth')
        net = Network_1Dconv().cpu()
        net2 = Network_div().cpu()
        
        net.load_state_dict(torch.load(weight_path2, map_location='cpu'))
        net2.load_state_dict(torch.load(weight_path2_div, map_location='cpu'))
        
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        opt2 = torch.optim.Adam(net2.parameters(), lr=0.001)
        
        net.to(hyperparams['device'])
        net2.to(hyperparams['device'])

    # train stage2 ##################################################################################
    retrain_stage2 = True
    
    if retrain_stage2 == True:
      print('train stage2...')

      slic_dataset = HyperX(img, train_gt, **hyperparams)
      slic_loader = data.DataLoader(slic_dataset, batch_size = 500000, shuffle=False)
      train_slic_gt = np.zeros_like(gt)
      for slic_batch_index, (img_batch, label_batch, indices) in enumerate(slic_loader):
            for i in indices:
                  row,col = i
                  train_slic_gt = find_slic(row,col,train_slic_gt)
      
      except_slic = np.where(train_slic_gt == 0)
      except_slic = [list(t) for t in zip(*except_slic)]
      except_train_gt = np.zeros_like(gt)
      for i in except_slic:
          r = i[0]
          c = i[1]
          except_train_gt[r][c] = gt[r][c]

      stage2_gt = except_train_gt.copy()
      #stage2_gt = test_gt.copy()
      tran_gt = train_gt.copy()
      
      hyperparams_semi=hyperparams.copy()
      hyperparams_semi['flip_augmentation']=True
      hyperparams_semi['radiation_augmentation']=True
      hyperparams_semi['mixture_augmentation']=True

      semi_loss_ratio=1.0
      semi_threshold=0.95

      epoch_2_start = 300
    
      for epoch_index in range(1):

        print('test...')
        torch.set_grad_enabled(False)
        net.eval()
        net2.eval()

        test_net1=True
        
        if test_net1==True:

            net.to(hyperparams['device'])
            net.eval()
            total_loss = []
            total_acc = 0
            total_sample = 0

            for test_batch_index, (img_batch, label_batch, indices) in enumerate(test_loader):
                label_batch = label_batch - 1

                img_batch = img_batch.to(hyperparams['device'])
                label_batch = label_batch.to(hyperparams['device'])

                predict = net(img_batch) 
                predict = predict.argmax(dim=1)
                acc = (predict == label_batch).sum()

                total_acc += acc
                total_sample += img_batch.size(0)

            mean_acc = total_acc.item() * 1.0 / total_sample

            print('[Test net1]  acc:%.4f%%\n'
              % ( mean_acc * 100))   

        st = time.time()

        total_loss_tran = []
       
        loss = 0
        loss_un= 0 
        loss_semi = 0
        loss1 = 10

        # make transductive dataset
        # 测试集得到每类好的样本
        tran_dataset = HyperX(img, stage2_gt, **hyperparams)
        tran_loader = data.DataLoader(tran_dataset, batch_size = 500000, shuffle=False)
     
        # Extract tran dataset and remove from stage2_gt
        total_acc_tran = 0
        total_sample_tran = 0

        # a for can test all
        for tran_batch_index, (img_batch, label_batch, indices) in enumerate(tran_loader):
            label_batch = label_batch-1

            img_batch = img_batch.to(hyperparams['device'])
            indices = indices.to(hyperparams['device'])
            label_batch = label_batch.to(hyperparams['device'])
            gt = gt.to(hyperparams['device'])

            predict = net(img_batch)
            predict_type = predict.argmax(dim=1)

            predict = F.softmax(predict, dim=1)
            max_pred, idx = predict.max(1)

            # 信息熵  
            info_entropy = -torch.sum(predict * torch.log2(predict),dim=-1)

            best_indices = dict()
            worst_indices = dict()
     
            for k in range(0,9):
                class_k_pre_idx = (predict_type == k).nonzero().squeeze(1)
                class_k_max_pred = max_pred[class_k_pre_idx]
                class_k_entropy = info_entropy[class_k_pre_idx]
                class_k_indices = indices[class_k_pre_idx]

                # find best per class
                class_k_best , class_k_best_idx = class_k_entropy.topk(100, dim=-1, largest=False, sorted=False)
                #class_k_best , class_k_best_idx = class_k_max_pred.topk(100, dim=-1, largest=True, sorted=False)

                # find worst per class
                class_k_worst , class_k_worst_idx = class_k_entropy.topk(100, dim=-1, largest=True, sorted=False)
                #class_k_worst , class_k_worst_idx = class_k_max_pred.topk(100, dim=-1, largest=False, sorted=False)

                class_k_best_indices = class_k_indices[class_k_best_idx]
                best_indices[k] = [i for i in class_k_best_indices]

                class_k_worst_indices = class_k_indices[class_k_worst_idx]
                worst_indices[k] = [i for i in class_k_worst_indices]
      
            # for a validation
            
            acc = (predict_type == label_batch).sum()
            total_acc_tran += acc
            total_sample_tran += img_batch.size(0)
        
        mean_acc_tran = total_acc_tran.item() * 1.0 / total_sample_tran

        print('acc:%.4f%%'% (mean_acc_tran * 100))

        # test the best selected
        selected_gt = np.zeros_like(gt.cpu())
        for index_1 in range(0,9):
            for index_2 in range(0,len(best_indices[index_1])):
                r,c=best_indices[index_1][index_2]
                selected_gt[r,c] = gt[r,c]

        tran_dataset = HyperX(img, selected_gt, **hyperparams)
        tran_loader = data.DataLoader(tran_dataset, batch_size=50000, shuffle=False)

        total_acc_tran = 0
        total_sample_tran = 0

        # a for can test all
        for tran_batch_index, (img_batch, label_batch, indices) in enumerate(tran_loader):
            label_batch = label_batch-1

            img_batch = img_batch.to(hyperparams['device'])
            indices = indices.to(hyperparams['device'])
            label_batch =  label_batch.to(hyperparams['device'])

            predict = net(img_batch)
            predict_type = predict.argmax(dim=1)

            acc = (predict_type == label_batch).sum()

            total_acc_tran += acc
            total_sample_tran += img_batch.size(0)
        
        mean_acc_tran = total_acc_tran.item() * 1.0 / total_sample_tran

        print('selected best acc:%.4f%%'% (mean_acc_tran * 100))

        # test the worst selected
        selected_gt = np.zeros_like(gt.cpu())
        for index_1 in range(0,9):
            for index_2 in range(0,len(worst_indices[index_1])):
                r,c=worst_indices[index_1][index_2]
                selected_gt[r,c] = gt[r,c]

        tran_dataset = HyperX(img, selected_gt, **hyperparams)
        tran_loader = data.DataLoader(tran_dataset, batch_size=50000, shuffle=False)

        total_acc_tran = 0
        total_sample_tran = 0

        # a for can test all
        for tran_batch_index, (img_batch, label_batch, indices) in enumerate(tran_loader):
            label_batch = label_batch-1

            img_batch = img_batch.to(hyperparams['device'])
            indices = indices.to(hyperparams['device'])
            label_batch =  label_batch.to(hyperparams['device'])

            predict = net(img_batch)
            predict_type = predict.argmax(dim=1)

            acc = (predict_type == label_batch).sum()

            total_acc_tran += acc
            total_sample_tran += img_batch.size(0)
        
        mean_acc_tran = total_acc_tran.item() * 1.0 / total_sample_tran

        print('selected worst acc:%.4f%%'% (mean_acc_tran * 100))

        # make the gt (add to train_gt)
        # 在训练集gt基础上加入选择的数据
     
        #slic区域获得best标签
        add_best = True
        if add_best == True:
         acc=0
         total_len=0
         for index_1 in range(0,9):
            for index_2 in range(0,len(best_indices[index_1])):
                r,c=best_indices[index_1][index_2]

                position = np.where(super_pixel == super_pixel[r, c])
                for i in range(0,len(position[0])):
                    x = position[0][i]
                    y = position[1][i]
                    
                    if gt[x,y] != 0:
                        tran_gt[x,y] = index_1+1
                        stage2_gt[x,y] = 0
                        total_len += 1

                        if tran_gt[x,y] == gt[x,y]:
                            acc += 1
                
         print('best slic acc: %.4f' % (acc/total_len))

        idx1 = tran_gt.nonzero()
        idx1 = idx1[0]
        print('size of tran_gt:%d' % len(idx1))

        # without aug dataset
        tran_dataset = HyperX(img, tran_gt, **hyperparams)
        tran_loader = data.DataLoader(tran_dataset, batch_size=50, shuffle=True)
   
        # Train trandataset
        # 选出的点和训练集数据一起做训练(伪标签)
        net = Network_1Dconv().cpu()             
        opt = torch.optim.Adam(net.parameters(), lr=0.001)     
        net.to(hyperparams['device'])

        torch.set_grad_enabled(True)
        net.train()

        loss_semi=0
        loss_low1=10

        print(epoch_2_start)
        for epoch2 in range(0,epoch_2_start):

         total_acc_tran = 0
         total_sample_tran = 0

         for tran_batch_index, (img_batch, label_batch, indices) in enumerate(tran_loader):

            img_batch_o = img_batch
            label_batch_o = label_batch
            label_batch_o = label_batch_o - 1 
            indices_o = indices

            net.to(hyperparams['device']) 

            # build up slic dataset (no slic+fixmatch)
            slic_gt=np.zeros_like(gt.cpu())

            for i in indices_o:
                  row,col = i
                  slic_gt = find_slic(row,col,slic_gt)

            slic_dataset = HyperX(img, slic_gt, **hyperparams_semi)
            slic_loader = data.DataLoader(slic_dataset, batch_size=5000, shuffle=True)             

            # a for can find all
            for tran_slic_index, (img_batch, label_batch, indices, img_batch_aug) in enumerate(slic_loader):
                        label_batch = label_batch - 1

                        img_batch_o = img_batch_o.to(hyperparams['device']) 
                        img_batch = img_batch.to(hyperparams['device'])
                        label_batch = label_batch.to(hyperparams['device'])
                        label_batch_o = label_batch_o.to(hyperparams['device'])
                        indices_o = indices_o.to(hyperparams['device'])
                        indices = indices.to(hyperparams['device'])
                        img_batch_aug = img_batch_aug.to(hyperparams['device'])
                        gt = gt.to(hyperparams['device'])

                        predict = net(img_batch_o)  
                        loss1 = loss_func(predict, label_batch_o)

                        # slic
                        predict_un = net(img_batch)  
                        predict_un = F.softmax(predict_un, dim=1)
                        loss_un = loss_func1(predict_un)     
                        loss_un = -torch.sum(torch.mean(predict_un * loss_un, dim=0, keepdim=True))

                        # semi 
                        # 使用slic作为数据增强数据集
                        if epoch2 > 30:

                          predict_type = predict_un.argmax(dim=1)
                          max_pred, idx = predict_un.max(1)
                          fit_pred_idx = (max_pred > semi_threshold).nonzero().squeeze(1)

                          fixmatch_img = img_batch_aug[fit_pred_idx]
                          fixmatch_label = predict_type[fit_pred_idx]
                          fixmatch_predict = net(fixmatch_img)
                          loss_semi = loss_func(fixmatch_predict , fixmatch_label)
                            
                        loss = loss1 + loss_un + loss_semi * semi_loss_ratio
                        #loss = loss1+loss_un

                        net.zero_grad()
                        loss.backward()
                        opt.step()

                        predict = predict.argmax(dim=1)
                        acc = (predict == label_batch_o).sum()

                        total_loss_tran.append(loss)
                        total_acc_tran += acc
                        total_sample_tran += img_batch_o.size(0)

                        if loss1 < loss_low1:
                            weight_path1 = os.path.join(weight_path, 'net.pth')
                            # print('Save Net weights to', weight_path1)
                            net_save = net.cpu()
                            torch.save(net_save.state_dict(), weight_path1)
                            loss_low1 = loss1

         mean_acc_tran = total_acc_tran.item() * 1.0 / total_sample_tran
         mean_loss_tran = sum(total_loss_tran) / total_loss_tran.__len__()

         print('[Train] epoch[%d/%d] acc:%.4f%% loss:%.4f loss1:%.4f loss_semi:%.4f'
              % (epoch_index, epoch2, mean_acc_tran * 100, mean_loss_tran.item(),  loss1 ,loss_semi))

         print('(LR:%f) Time of a epoch:%.4fs\n' % (opt.param_groups[0]['lr'], time.time() - st))

        #epoch_2_start += 50

        weight_path2 = os.path.join(weight_path, 'net.pth')
        net = Network_1Dconv().cpu()
        
        net.load_state_dict(torch.load(weight_path2, map_location='cpu'))
        
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        
        net.to(hyperparams['device'])

    # test net1###################################################################################
    print('test...')
    torch.set_grad_enabled(False)

    test_net1=True
        
    if test_net1==True:

        net.to(hyperparams['device'])
        net.eval()
        total_loss = []
        total_acc = 0
        total_sample = 0

        for test_batch_index, (img_batch, label_batch, indices) in enumerate(test_loader):
            label_batch = label_batch - 1

            img_batch = img_batch.to(hyperparams['device'])
            label_batch = label_batch.to(hyperparams['device'])

            predict = net(img_batch) 
            predict = predict.argmax(dim=1)
            acc = (predict == label_batch).sum()

            total_acc += acc
            total_sample += img_batch.size(0)

        mean_acc = total_acc.item() * 1.0 / total_sample

        print('[Test net1]  acc:%.4f%%\n'
              % ( mean_acc * 100))   
    # test all###################################################################################
    net.to(hyperparams['device'])
    net.eval()
    net2.to(hyperparams['device'])
    net2.eval()
    total_loss = []
    total_acc = 0
    total_sample = 0
    total_predicts = []
    total_labels = []
    total_indices = []

    for test_batch_index, (img_batch, label_batch, indices) in enumerate(test_loader):
        img_batch_o = img_batch
        label_batch_o = label_batch
        label_batch_o = label_batch_o - 1
        indices_o = indices

        img_batch_o = img_batch_o.to(hyperparams['device'])
        img_batch = img_batch.to(hyperparams['device'])
        label_batch = label_batch.to(hyperparams['device'])
        label_batch_o = label_batch_o.to(hyperparams['device'])
        indices_o = indices_o.to(hyperparams['device'])
        indices = indices.to(hyperparams['device'])
        gt = gt.to(hyperparams['device'])

        for i in indices_o:
                        row,col = i
                        slic_gt = np.zeros_like(gt.cpu())                      
                        slic_gt = find_slic(row,col,slic_gt)
                        slic_dataset = HyperX(img, slic_gt, **hyperparams)
                        slic_loader = data.DataLoader(slic_dataset, batch_size=5000, shuffle=False) 

                        for test_slic_index, (slic_img_batch, slic_label_batch, slic_indices) in enumerate(slic_loader):

                            slic_img_batch = slic_img_batch.to(hyperparams['device'])
                            slic_label_batch = slic_label_batch.to(hyperparams['device'])
                            slic_indices = slic_indices.to(hyperparams['device'])

                            predict_cnn = net(slic_img_batch)
                            predict , mark = net2(predict_cnn, slic_indices, row, col)
                            predict=predict[mark]
                            slic_label =  slic_label_batch[mark:mark+1]-1
                            
                            loss = loss_func(predict, slic_label)

                            predict = predict.argmax(dim=1)
                            acc = (predict == slic_label).sum()
                            predict_1 = predict.cpu().numpy().tolist()
                            label_batch_1 = slic_label.cpu().numpy().tolist()
                            total_predicts.append(predict_1)
                            total_labels.append(label_batch_1)
                            total_loss.append(loss)
                            total_acc += acc
                            
        total_sample += img_batch_o.size(0)

        indices_o = indices_o.cpu().numpy().tolist()
        total_indices.append(indices_o)

    mean_acc = total_acc.item() * 1.0 / total_sample
    mean_loss = sum(total_loss) / total_loss.__len__()

    print('[Test]  acc:%.4f%% loss:%.4f\n'
              % ( mean_acc * 100, mean_loss.item()))


    sio.savemat(os.path.join(result_path, 'pre_label.mat'), {'pre_label': total_predicts})
    sio.savemat(os.path.join(result_path, 'test_label.mat'), {'test_label': total_labels})
    sio.savemat(os.path.join(result_path, 'total_indices.mat'), {'total_indices': total_indices})

    # aa oa kappa ################################################################################
    class_num = np.max(train_gt)
    matrix = np.zeros((class_num,class_num),dtype=np.int64)
    for i in range(total_predicts.__len__()):
        l = total_predicts[i].__len__()
        for j in range(l):
            matrix[total_predicts[i][j], total_labels[i][j]] += 1

    ac_list = []
    for i in range(len(matrix)):
        ac = matrix[i, i] / sum(matrix[:, i])
        ac_list.append(ac)
        print(i + 1, 'class:', '(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
    print('confusion matrix:')
    print(np.int_(matrix))
    print('total right num:', np.sum(np.trace(matrix)))
    accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
    print('oa:', accuracy)
    # kappa
    kk = 0
    for i in range(matrix.shape[0]):
        kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
    pe = kk / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    ac_list = np.asarray(ac_list)
    aa = np.mean(ac_list)
    oa = accuracy
    print('aa:', aa)
    print('kappa:', kappa)

    sio.savemat(os.path.join(result_path, 'result.mat'),{'oa': oa, 'aa': aa, 'kappa': kappa, 'ac_list': ac_list, 'matrix': matrix})
