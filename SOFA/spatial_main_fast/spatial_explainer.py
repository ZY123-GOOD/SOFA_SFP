from __future__ import print_function, division
import torch
import numpy as np
from utils.shap_utils import *



def spatial_shap(model,img_numpy,image_class,concept_masks,fc,feat_exp,image_norm=None,lr=0.008, epochs=4,mc=2000):
    
    
    net = learn_PIE(feat_exp,model,concept_masks,img_numpy,image_class,fc,lr=lr,epochs=epochs,image_norm=image_norm) ### learning the PIE module 
    feat_num = len(concept_masks)
    shap_val = []
    mc = mc
    
    for i in range(feat_num):

        bin_x_tmp = torch.bernoulli(torch.full((mc, feat_num), 0.5)) # 生成随机二项分布的张量
        bin_x_tmp_sec = bin_x_tmp.clone()  # 复制张量


        bin_x_tmp[:, i] = 1
        bin_x_tmp_sec[:, i] = 0
        
        pre_shap = (feat_prob(fc,net.forward_feat(bin_x_tmp.cuda()),image_class) - feat_prob(fc,net.forward_feat(bin_x_tmp_sec.cuda()),image_class)).detach().cpu().numpy()

        shap_val.append(pre_shap.sum()/mc)
        # print(bin_x_tmp.shape)
        # break
    ans = shap_val.index(max(shap_val))
    shap_list = shap_val
    
    shap_list = np.array(shap_list)
    shap_arg = np.argsort(-shap_list)
    # print(ans,max(shap_val))
    auc_mask = np.expand_dims(np.array([concept_masks[shap_arg[:i+1]].sum(0) for i in range(len(shap_arg))]).astype(bool),3)


    return auc_mask, shap_list,shap_arg