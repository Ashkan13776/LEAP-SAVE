import torch
from torch import nn
from torch.nn import ReLU, Linear
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR,LambdaLR,MultiStepLR

import torch_geometric as tg
import torch_geometric.nn as tg_nn
import torch_geometric.utils as tg_utils
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.init as init

from train_utils import cuda2numpy,df2cuda
from train_utils import deformation_capture_avg,deformation_capture_max,deformation_capture_median
from train_utils import standard_scaler_fit,standard_scaler_transform,invers_scale_transform,feature_transform_regularizer

from pointnet import PointNetfeat
from data_preprocessing import preprocess

import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import json
import time
import sys
import os

from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import matplotlib.pyplot as plt
import random

num_points=7232

def train_test(num_epochs=1000,lr=5e-4,weight_decay=5e-4,opt='adam',milestones=[150,300,600],gamma=0.1,split=0.8,batch_size=1,device='cuda:0'):
    
    data_batch,mean_x,mean_y,std_x,std_y = preprocess(device=device)
    
    model=PointNetfeat()
    model=model.to(device)
    
    criterion = nn.MSELoss()

    params=model.parameters()
    
    num_epochs=num_epochs
    
    if opt=='adam':

        optimizer = Adam
        optim = optimizer(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer= SGD
        optim = optimizer(params,lr=lr,weight_decay=weight_decay)
        
    scheduler = MultiStepLR(optim, milestones=milestones, gamma=gamma)

    epochMetrics = defaultdict(list)
    split_idx = int(split * len(data_batch))
    train_data = data_batch[:split_idx]
    test_data = data_batch[split_idx:]
    loader_train = DataLoader(train_data, batch_size=1, shuffle=True)
    loader_test = DataLoader(test_data, batch_size=1, shuffle=False)
    #training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        phase='train'

        log_capture_def_avg=[]
        log_capture_def_max=[]
        log_capture_def_median=[]
        log_max_mag_err=[]

        for data in loader_train:

            data=data.to(device)

            data.x[:,:6]=standard_scaler_transform(data.x[:,:6],mean_x,std_x)
            data.y=standard_scaler_transform(data.y,mean_y,std_y)
            data.x = data.x.reshape(1,num_points,7)
            # Forward pass
            y_pred,_,trans_feat = model(data.x.float())
            y=data.y.squeeze()
            y_pred=y_pred.squeeze(0)
            # Compute the loss
            loss = criterion(y_pred.reshape(num_points,3).float(), y.float())+0.01*feature_transform_regularizer(trans_feat,device=device)

            # Backpropagation and optimization
            optim.zero_grad()
            loss.backward()
            optim.step()

            #metrics

            y_inv=invers_scale_transform(y,mean_y,std_y)
            y_pred_inv=invers_scale_transform(y_pred,mean_y,std_y)

            error_vectors = torch.abs(y_pred_inv - y_inv)
            error_magnitudes = torch.sqrt(torch.sum(error_vectors ** 2, dim=1))

            abs_gt = torch.sqrt(torch.sum(torch.abs(data.y.squeeze())**2,dim=-1))

            mag_err=torch.max(error_magnitudes)
            capture_def_avg=deformation_capture_avg(torch.mean(error_magnitudes**2),abs_gt)
            capture_def_max=deformation_capture_max(torch.mean(error_magnitudes**2),abs_gt)
            capture_def_median=deformation_capture_median(torch.mean(error_magnitudes**2),abs_gt)


            log_capture_def_avg.append(capture_def_avg.item())
            log_capture_def_max.append(capture_def_max.item())
            log_capture_def_median.append(capture_def_median.item())
            log_max_mag_err.append(mag_err.item())


        mean_log_capture_def_avg=np.mean(log_capture_def_avg)
        mean_log_capture_def_max=np.mean(log_capture_def_max)
        mean_log_capture_def_median=np.mean(log_capture_def_median)
        mean_log_max_mag_err=np.mean(log_max_mag_err)



        epochMetrics[f'{phase}_def_avg'].append(mean_log_capture_def_avg)
        epochMetrics[f'{phase}_def_max'].append(mean_log_capture_def_max)
        epochMetrics[f'{phase}_def_median'].append(mean_log_capture_def_median)
        epochMetrics[f'{phase}_max_mag_err'].append(mean_log_max_mag_err)


        # Testing loop
        model.eval()  # Set the model to evaluation mode
        phase='test'


        log_capture_def_avg=[]
        log_capture_def_max=[]
        log_capture_def_median=[]
        log_max_mag_err=[]

        with torch.no_grad():
            for data in loader_test:

                data=data.to(device)
                data.x[:,:6]=standard_scaler_transform(data.x[:,:6],mean_x,std_x)

                data.y=standard_scaler_transform(data.y,mean_y,std_y)

                data.x = data.x.reshape(1,num_points,7)
                # Forward pass
                y_pred,_,_ = model(data.x.float())

                y=data.y.squeeze()
                y_pred=y_pred.squeeze(0)

                loss=criterion(y,y_pred)

                y_inv=invers_scale_transform(y,mean_y,std_y)
                y_pred_inv=invers_scale_transform(y_pred,mean_y,std_y)

                abs_gt = torch.sqrt(torch.sum(torch.abs(data.y.squeeze())**2,dim=-1))

                error_vectors = torch.abs(y_pred_inv - y_inv)
                error_magnitudes = torch.sqrt(torch.sum(error_vectors ** 2, dim=1))
                mag_err=torch.max(error_magnitudes)

                capture_def_avg=deformation_capture_avg(torch.mean(error_magnitudes**2),abs_gt)
                capture_def_max=deformation_capture_max(torch.mean(error_magnitudes**2),abs_gt)
                capture_def_median=deformation_capture_median(torch.mean(error_magnitudes**2),abs_gt)

                log_capture_def_avg.append(capture_def_avg.item())
                log_capture_def_max.append(capture_def_max.item())
                log_capture_def_median.append(capture_def_median.item())
                log_max_mag_err.append(mag_err.item())


        mean_log_capture_def_avg=np.mean(log_capture_def_avg)
        mean_log_capture_def_max=np.mean(log_capture_def_max)
        mean_log_capture_def_median=np.mean(log_capture_def_median)
        mean_log_max_mag_err=np.mean(log_max_mag_err)


        epochMetrics[f'{phase}_def_avg'].append(mean_log_capture_def_avg)
        epochMetrics[f'{phase}_def_max'].append(mean_log_capture_def_max)
        epochMetrics[f'{phase}_def_median'].append(mean_log_capture_def_median)
        epochMetrics[f'{phase}_max_mag_err'].append(mean_log_max_mag_err)

        scheduler.step()

        if epoch%10==0:

            train_def_avg = epochMetrics['train_def_avg'][-1]  
            test_def_avg = epochMetrics['test_def_avg'][-1]    
            print(f"Def Avg. error [train]: {train_def_avg:.2f} %")
            print(f"Def Avg. error [test]: {test_def_avg:.2f} %")

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, 'first_run_states.pth')
    
    
if __name__== '__main__':
    
    train_test()
    
    
    
    
    
    
    

