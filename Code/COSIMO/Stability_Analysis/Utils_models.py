"""
COSIMO model together with ablation study 
"""

import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import torchmetrics
import torch.nn as nn

from utils_layers import sccnn_conv_cont
'''for ablations below'''
# from model.sccnn_conv import sccnn_conv_no_n_to_e,sccnn_conv_no_e_to_e,sccnn_conv_no_t_to_e,sccnn_conv_no_n_to_n

# from model.sccnn_conv import sccnn_conv_no_n_to_e_sc_1,sccnn_conv_no_e_to_e_sc_1,sccnn_conv_no_e_to_n_sc_1,sccnn_conv_no_n_to_n_sc_1


class COSIMO(nn.Module):
    def __init__(self, F_in, F_intermediate, F_out, b1, b2, evals_0, evecs_0, evals_1l, evecs_1l, evals_1u,
                 evecs_1u, evals_2, evecs_2, sigma, model_name, single_t):
        super(COSIMO, self).__init__()
        self.num_features = [F_in] + [F_intermediate[l] for l in range(len(F_intermediate))] + [F_out] # number of features vector e.g., [1 5 5 5 1]
        self.num_layers = len(self.num_features) 
        self.b1 = b1
        self.b2 = b2 
        self.evals_0 = evals_0
        self.evecs_0 = evecs_0
        self.evals_1l = evals_1l
        self.evecs_1l = evecs_1l
        self.evals_1u = evals_1u
        self.evecs_1u = evecs_1u
        self.evals_2 = evals_2
        self.evecs_2 = evecs_2
        self.single_t = single_t
        
        self.sigma = sigma
        nn_layer = []
        print(model_name)
        
        for l in range(self.num_layers-1):
            hyperparameters = {"F_in":self.num_features[l],"F_out":self.num_features[l+1],"b1":self.b1, "b2":self.b2, 
                               "evals_0":self.evals_0, "evecs_0":self.evecs_0,
                               "evals_1l":self.evals_1l, "evecs_1l":self.evecs_1l,
                               "evals_1u":self.evals_1u, "evecs_1u":self.evecs_1u,
                               "evals_2":self.evals_2, "evecs_2":self.evecs_2,
                               "sigma":self.sigma, 'single_t': self.single_t}
            if model_name in ['sccnn_node','sccnn_edge','sccnn_node_missing_node','sccnn_node_missing_edge','sccnn_node_missing_node_edge']: # original sccnn for sc of order two, and some ablation study: limited input data 
                nn_layer.extend([sccnn_conv_cont(**hyperparameters)]) 
            # ### ablation for sc order 2 
            # elif model_name in ['sccnn_node_no_t_to_e']: # no triangle to edge contribution
            #     nn_layer.extend([sccnn_conv_no_t_to_e(**hyperparameters)])  
            # elif model_name in ['sccnn_node_no_n_to_e']: # no node to edge
            #     nn_layer.extend([sccnn_conv_no_n_to_e(**hyperparameters)])  
            # elif model_name in ['sccnn_node_no_e_to_e']: # no edge to edge
            #     nn_layer.extend([sccnn_conv_no_e_to_e(**hyperparameters)])  
            # elif model_name in ['sccnn_node_no_n_to_n']: # no node to node
            #     nn_layer.extend([sccnn_conv_no_n_to_n(**hyperparameters)])   
            # ##### ablation for sc order 1    
            # elif model_name in ['sccnn_node_no_n_to_n_sc_1']: # no b2, no node to node 
            #     nn_layer.extend([sccnn_conv_no_n_to_n_sc_1(**hyperparameters)]) 
            # elif model_name in ['sccnn_node_no_e_to_e_sc_1']: # no b2, no edge to edge
            #     nn_layer.extend([sccnn_conv_no_e_to_e_sc_1(**hyperparameters)])
            # elif model_name in ['sccnn_node_no_e_to_n_sc_1']: # no b2, no edge to node
            #     nn_layer.extend([sccnn_conv_no_e_to_n_sc_1(**hyperparameters)]) 
            # elif model_name in ['sccnn_node_no_n_to_e_sc_1']: # no b2, no node to edge
            #     nn_layer.extend([sccnn_conv_no_n_to_e_sc_1(**hyperparameters)])  
            else: 
                raise Exception('invalid model type')
        
        self.simplicial_nn = nn.Sequential(*nn_layer)

    def forward(self,x):
        return self.simplicial_nn(x)#.view(-1,1).T