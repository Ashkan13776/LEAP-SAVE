import torch
import os

def preprocess(device,num_points=7232,data_path='7k_mesh_data'):
    
    num_points=7232
    data_list=os.listdir(data_path)
    data_batch=[]
    for i,file in enumerate(data_list):
        data=torch.load(os.path.join(data_path, 'data{}.pt'.format(i)))
        data_batch.append(data)
    # Stack all node features into a single tensor
    node_features = torch.cat([graph.x[:,:6] for graph in data_batch], dim=0)

    # Calculate the mean and standard deviation across all samples
    mean_x = torch.mean(node_features,dim=0)
    std_x = torch.std(node_features,dim=0)
    # Stack all node features into a single tensor
    node_targets = torch.cat([graph.y for graph in data_batch], dim=0)

    # Calculate the mean and standard deviation across all samples
    mean_y = torch.mean(node_targets,dim=0)
    std_y = torch.std(node_targets,dim=0)
    mean_x,std_x,mean_y,std_y = mean_x.to(device),std_x.to(device),mean_y.to(device),std_y.to(device)
    return data_batch,mean_x,mean_y,std_x,std_y