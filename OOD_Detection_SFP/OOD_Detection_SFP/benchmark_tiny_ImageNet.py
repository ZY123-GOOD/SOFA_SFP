#!/usr/bin/env python
import argparse
import torch
import numpy as np
from tqdm import tqdm
import mmcv
from numpy.linalg import norm, pinv
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import pandas as pd
import os
import faiss
import torch.nn as nn 
import random
import timm
# from easyrobust import models
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from dataset_twoinputs import DatasetFolder_two_tiny
from test_utils import get_measures

from mylogger import *
from PIL import Image


class predict_model(nn.Module): ### a simple NN network
    def __init__(self,head,mnorm,flatten):
        super(predict_model, self).__init__()
        self.fc = head
        self.norm = mnorm
        self.flatten = flatten

    def forward(self,x):
        x = x.reshape([x.shape[0],x.shape[1],x.shape[2]*x.shape[3]])
        out = torch.mean(x,dim=2)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        
        out = self.norm(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def forward_threshold(self,x, threshold=1.0):
        x = x.reshape([x.shape[0],x.shape[1],x.shape[2]*x.shape[3]])
        out = torch.mean(x,dim=2)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        
        out = self.norm(out)
        out = self.flatten(out)
        out = out.clip(max=threshold)
        out = self.fc(out)
        return out

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
setup_seed(1)



def main():
    DEBUG_FLAG=False
    OOD_NUM = 4
    Temp = 5
        
    model = timm.create_model('tiny_vit_21m_224.in1k', checkpoint_path="shared_models/tiny_vit_21m_224.in1k.bin")

    model = model.cuda()
    model = model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    mnorm = model.head.norm
    flatten = model.head.flatten
    prem = predict_model(model.head.fc,mnorm,flatten)
    
    ### ID dataloader
    val_image_dir = "./datasets/images/val"
    val_set = DatasetFolder_two_tiny(val_image_dir, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    ### OOD dataloader
    ood_name = ['SUN', 'Places','dtd','iNaturalist']
    ood_loader_list = []
    
    ood_image_dir = "./datasets/images/SUN"
    ood_set = DatasetFolder_two_tiny(ood_image_dir, transform = transform)
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=1, shuffle=True,num_workers=4, pin_memory=True, drop_last=False)
    ood_loader_list.append(ood_loader)   
    
    ood_image_dir = "./datasets/images/Places"
    ood_set = DatasetFolder_two_tiny(ood_image_dir, transform = transform)
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=1, shuffle=True,num_workers=4, pin_memory=True, drop_last=False)
    ood_loader_list.append(ood_loader)    
    
    ood_image_dir = "./datasets/images/dtd"
    ood_set = DatasetFolder_two_tiny(ood_image_dir, transform = transform)
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=1, shuffle=True,num_workers=4, pin_memory=True, drop_last=False)
    ood_loader_list.append(ood_loader)

    ood_image_dir = "./datasets/images/iNaturalist"
    ood_set = DatasetFolder_two_tiny(ood_image_dir, transform = transform)
    ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=1, shuffle=True,num_workers=4, pin_memory=True, drop_last=False)
    ood_loader_list.append(ood_loader)
        
    logpath='./logs/'+'vit'
    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    make_print_to_file(path=logpath, subname='')
    
    print(f"ood datasets: {ood_name}")

    purified_num = 15    
    index = 49-purified_num
    
# #####################---------------------------------------
    
    method = 'Energy + SFP'
    print(f'\n{method}')
    result = []
    id_scores = []
    model = model.eval()
    iters = 0
    for image, image_label, shapleyargs in tqdm(val_loader):  
        if len(shapleyargs[0])==1:
            # nonshape = nonshape + 1
            continue
        iters = iters+1
        if iters>500 and DEBUG_FLAG==True:
            break
        ft = model.forward_features(image.cuda())
        re_ft = ft.reshape([576,49])
        tmp_ft = re_ft[:,shapleyargs[0][:index]]
        tmp_ft = torch.unsqueeze(tmp_ft,dim=2)
        tmp_ft = torch.unsqueeze(tmp_ft,dim=0)
        score1 = torch.logsumexp(prem(tmp_ft) / 1, dim=1).cpu().detach()[0]
        id_scores.append(score1)
    id_scores = np.array(id_scores).reshape((-1, 1))

    for i_ood in range(OOD_NUM):
        model = model.eval()
        ood_scores = []
        iters = 0
        for image, image_label, shapleyargs in tqdm(ood_loader_list[i_ood]):  
            if len(shapleyargs[0])==1:
                # nonshape = nonshape + 1
                continue
            iters = iters+1
            if iters>500 and DEBUG_FLAG==True:
                break
            ft = model.forward_features(image.cuda())
            re_ft = ft.reshape([576,49])
            tmp_ft = re_ft[:,shapleyargs[0][:index]]
            tmp_ft = torch.unsqueeze(tmp_ft,dim=2)
            tmp_ft = torch.unsqueeze(tmp_ft,dim=0)
            score1 = torch.logsumexp(prem(tmp_ft) / 1, dim=1).cpu().detach()[0]
            ood_scores.append(score1)
        ood_scores = np.array(ood_scores).reshape((-1, 1))
        
        auc_ood,_,_,fpr_ood = get_measures(id_scores, ood_scores)
        
        result.append(dict(method=method, oodset=ood_name[i_ood], auroc=auc_ood, fpr=fpr_ood))
        print(f'{method} {index} index: {ood_name[i_ood]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')    



    
    
    purified_num = 0    
    index = 49-purified_num
    
# #####################---------------------------------------
    
    method = 'Vanilla Energy'
    print(f'\n{method}')
    result = []
    id_scores = []
    model = model.eval()
    iters = 0
    for image, image_label, shapleyargs in tqdm(val_loader):  
        if len(shapleyargs[0])==1:
            # nonshape = nonshape + 1
            continue
        iters = iters+1
        if iters>500 and DEBUG_FLAG==True:
            break
        ft = model.forward_features(image.cuda())
        re_ft = ft.reshape([576,49])
        tmp_ft = re_ft[:,shapleyargs[0][:index]]
        tmp_ft = torch.unsqueeze(tmp_ft,dim=2)
        tmp_ft = torch.unsqueeze(tmp_ft,dim=0)
        score1 = torch.logsumexp(prem(tmp_ft) / 1, dim=1).cpu().detach()[0]
        id_scores.append(score1)
    id_scores = np.array(id_scores).reshape((-1, 1))

    for i_ood in range(OOD_NUM):
        model = model.eval()
        ood_scores = []
        iters = 0
        for image, image_label, shapleyargs in tqdm(ood_loader_list[i_ood]):  
            if len(shapleyargs[0])==1:
                # nonshape = nonshape + 1
                continue
            iters = iters+1
            if iters>500 and DEBUG_FLAG==True:
                break
            ft = model.forward_features(image.cuda())
            re_ft = ft.reshape([576,49])
            tmp_ft = re_ft[:,shapleyargs[0][:index]]
            tmp_ft = torch.unsqueeze(tmp_ft,dim=2)
            tmp_ft = torch.unsqueeze(tmp_ft,dim=0)
            score1 = torch.logsumexp(prem(tmp_ft) / 1, dim=1).cpu().detach()[0]
            ood_scores.append(score1)
        ood_scores = np.array(ood_scores).reshape((-1, 1))
        
        auc_ood,_,_,fpr_ood = get_measures(id_scores, ood_scores)
        
        result.append(dict(method=method, oodset=ood_name[i_ood], auroc=auc_ood, fpr=fpr_ood))
        print(f'{method} {index} index: {ood_name[i_ood]} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')    
        

if __name__ == '__main__':
    main()