import torch
import torch.nn as nn


def cuda2numpy(Tensor, device):
    return Tensor.detach().cpu().numpy()

def df2cuda(df):
    return torch.from_numpy(np.array(df)).to(device)

def deformation_capture_avg(error, abs_gt):
    capture = ((error)/torch.mean(abs_gt))
    return capture

def deformation_capture_max(error, abs_gt):
    capture = ((error)/torch.max(abs_gt))
    return capture

def deformation_capture_median(error, abs_gt):
    capture = ((error)/torch.median(abs_gt))
    return capture

def standard_scaler_fit(x):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)
    return mean, std

def standard_scaler_transform(x,mean=None,std=None):

    scaled_x = (x - mean) / (std + 1e-8)
    return scaled_x

def invers_scale_transform(x,mean,std):
    inverse_scaled = x*std + mean
    return inverse_scaled

def feature_transform_regularizer(trans, device):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :].to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss