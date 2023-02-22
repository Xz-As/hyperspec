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

from lib.GCN_funcs import aff_to_adj , kCenterGreedy , MultiCEFocalLoss
from lib.GCN import GCN

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]
#dataset_names = 'IndianPines'
dataset_names = 'PaviaU'
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
group_dataset.add_argument('--training_sample', type=float, default=10, help="Percentage of samples to use for training (default: 10%)")
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
 
# BCE loss
def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss

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

  new_gt = train_gt.copy()

  loss_func2 = MultiCEFocalLoss(9)
 
  for rounds in range(1):
    i = str(time_index)
    weight_path = os.path.join('weights', i)
    result_path = os.path.join('results', i)
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    # read net1
    weight_path2 = os.path.join(weight_path, 'net.pth')
    net = Network_1Dconv().cpu()
    net.load_state_dict(torch.load(weight_path2, map_location='cpu'))        
    net.to(hyperparams['device'])
    torch.set_grad_enabled(False)
    net.eval()

    # test
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

            predict,_ = net(img_batch) 
            predict = predict.argmax(dim=1)
            acc = (predict == label_batch).sum()

            total_acc += acc
            total_sample += img_batch.size(0)

        mean_acc = total_acc.item() * 1.0 / total_sample

        print('[Test net1]  acc:%.4f%%\n'
              % ( mean_acc * 100))

    # get features from all datas
    feature_dataset = HyperX(img, gt, **hyperparams)
    feature_loader = data.DataLoader(feature_dataset, batch_size=100000,shuffle=False)

      # a for can find all 
    for batch_index, (img_batch, label_batch, indices) in enumerate(feature_loader): 
            img_batch = img_batch.to(hyperparams['device'])
            _ , feature = net(img_batch)
            img_batch = img_batch.cpu()
    
    # find labeled/unlabeled points
    feature_dataset = HyperX(img, new_gt, **hyperparams)
    feature_loader = data.DataLoader(feature_dataset, batch_size=100000,shuffle=False)
    for _, (_, _, labeled_indices) in enumerate(feature_loader):
                continue
    
    indices = indices.numpy().tolist()
    labeled_indices = labeled_indices.numpy().tolist()
    labeled_pos = []
    for i in labeled_indices:
      labeled_pos.append(indices.index(i))
    unlabeled_pos = [x for x in range(len(indices)) if x not in labeled_pos]
   
    # normalize
    feature = nn.functional.normalize(feature)
    adj = aff_to_adj(feature)

    # build GCN
    gcn_module = GCN(nfeat=feature.shape[1],
                         nhid=128,
                         nclass=1,
                         dropout=0.3).cuda()
    
    models = {'gcn_module': gcn_module}
    optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=1e-3, weight_decay=5e-4)
    optimizers = {'gcn_module': optim_backbone}

    labeled_pos = np.array(labeled_pos) # 需要修改为每轮更新形式
    unlabeled_pos = np.array(unlabeled_pos)

    # train GCN
    print('training GCN...')

    torch.set_grad_enabled(True)
    models['gcn_module'].train()

    #1:labeled 0:unlabeled
    binary_label = torch.zeros([len(indices)])
    binary_label[labeled_pos] = 1
    binary_label = binary_label.long()

    for _ in range(200):
            st = time.time()

            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](feature, adj)
            lamda = 1.2
            loss = BCEAdjLoss(outputs, labeled_pos, unlabeled_pos, lamda)
            loss.backward()
            optimizers['gcn_module'].step()

            outputs = outputs.squeeze(-1)

    print('GCN finished')

    # get coreset distance
    CUDA_VISIBLE_DEVICES = 0
    ADDENDUM = 90

    models['gcn_module'].eval()
    with torch.no_grad():
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = feature.cuda()
        scores, _, feat = models['gcn_module'](inputs, adj)
            
        feat = feat.detach().cpu().numpy()
        new_av_idx = labeled_pos # 需要修改为每轮更新形式
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        batch = torch.Tensor(batch).long()
        indices = torch.Tensor(indices).long()
        new_indices = indices[batch]

    # add to gt # 需修改为每轮更新形式
    ones_in_train = 0
    for i in new_indices:
        r,c = i

        if train_gt[r,c] != 0:
            ones_in_train +=1

        new_gt[r,c] = gt[r,c]

    print('ones in train: %d' % (ones_in_train))
    idx1 = new_gt.nonzero()
    idx1 = idx1[0]
    print('size of new_gt:%d' % len(idx1))


  expend = True

  if expend == True:
    
    # expend train_gt with slic
    indices = new_gt.nonzero()
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

            if new_gt[x,y]!=0:
                selected+=1
        
            if(gt[x,y]!=0):
                new_gt[x,y] = gt[r,c]

            if new_gt[x,y] == gt[x,y]:
                acc+=1
         
    print('slic acc: %.4f' % (acc*1.0/total_len))

    #train
    new_dataset = HyperX(img, new_gt, **hyperparams)
    new_loader = data.DataLoader(new_dataset, batch_size=64,shuffle=True)

    net = Network_1Dconv().cpu()             
    opt = torch.optim.Adam(net.parameters(), lr=0.001)     
    sched = lr_scheduler.MultiStepLR(opt, milestones=[300], gamma=1)
    net.to(hyperparams['device'])

    torch.set_grad_enabled(True)
    net.train()

    loss_semi=0
    loss_low1=10
    loss_low2 = 10
    loss1, loss_un, loss2 = 10, 10, 10

    total_loss_tran = []

    loss_func = nn.CrossEntropyLoss()
    loss_func1 = nn.LogSoftmax(dim=1)
    
    hyperparams_semi=hyperparams.copy()
    hyperparams_semi['flip_augmentation']=True
    hyperparams_semi['radiation_augmentation']=True
    hyperparams_semi['mixture_augmentation']=True

    epoch_2_start = 300

    semi_loss_ratio=1.0
    semi_threshold=0.95

    for epoch2 in range(0,epoch_2_start):

         total_acc_tran = 0
         total_sample_tran = 0

         for _, (img_batch, label_batch, indices) in enumerate(new_loader):

            img_batch_o = img_batch
            label_batch_o = label_batch
            label_batch_o = label_batch_o - 1 
            indices_o = indices

            net.to(hyperparams['device']) 

            # build up slic dataset 
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

                        predict,feature3 = net(img_batch_o)  
                        loss1 = loss_func(predict, label_batch_o)
                        #loss1 = loss_func2(predict, label_batch_o)

                        # slic
                        predict_un,feature3 = net(img_batch)  
                        predict_un = F.softmax(predict_un, dim=1)
                        loss_un = loss_func1(predict_un)     
                        loss_un = -torch.sum(torch.mean(predict_un * loss_un, dim=0, keepdim=True))

                        # semi 
                        # 使用slic作为数据增强数据集
                        if epoch2 > 1000:

                          predict_type = predict_un.argmax(dim=1)
                          max_pred, idx = predict_un.max(1)
                          fit_pred_idx = (max_pred > semi_threshold).nonzero().squeeze(1)

                          fixmatch_img = img_batch_aug[fit_pred_idx]
                          fixmatch_label = predict_type[fit_pred_idx]
                          fixmatch_predict,feature3 = net(fixmatch_img)
                          loss_semi = loss_func(fixmatch_predict , fixmatch_label)
                            
                        loss = loss1 + loss_un + loss_semi * semi_loss_ratio

                        net.zero_grad()
                        loss.backward()
                        opt.step()
                        sched.step()

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

         print('[Train] epoch[%d] acc:%.4f%% loss:%.4f loss1:%.4f loss_semi:%.4f'
              % (epoch2, mean_acc_tran * 100, mean_loss_tran.item(),  loss1 ,loss_semi))

         print('(LR:%f) ' % (opt.param_groups[0]['lr']))
    
    # read net1
    weight_path2 = os.path.join(weight_path, 'net.pth')
    net = Network_1Dconv().cpu()        
    net.load_state_dict(torch.load(weight_path2, map_location='cpu'))        
    opt = torch.optim.Adam(net.parameters(), lr=0.001)        
    net.to(hyperparams['device'])

    # test
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

            predict,_ = net(img_batch) 
            predict = predict.argmax(dim=1)
            acc = (predict == label_batch).sum()

            total_acc += acc
            total_sample += img_batch.size(0)

        mean_acc = total_acc.item() * 1.0 / total_sample

        print('[Test net1]  acc:%.4f%%\n'
              % ( mean_acc * 100))

  