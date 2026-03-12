"""SCCNN implementation for complex classification."""
import torch
import random
from itertools import product

import networkx as nx
import numpy as np
import toponetx as tnx
import torch
from scipy.spatial import Delaunay, distance
from torch import nn
from torch.utils.data.dataset import Dataset

from topomodelx.nn.simplicial.scone_layer import SCoNeLayer
from Utils.sccnn_exp_layer import SCCNNLayer_exp, SCCNN_Exp_Layer_2


class COSIMO(torch.nn.Module):
    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = torch.nn.Linear(in_channels_all[0], hidden_channels_all[0])
        self.in_linear_1 = torch.nn.Linear(in_channels_all[1], hidden_channels_all[1])
        self.in_linear_2 = torch.nn.Linear(in_channels_all[2], hidden_channels_all[2])

        self.layers = torch.nn.ModuleList(
            SCCNNLayer_exp(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                conv_order=conv_order,
                sc_order=sc_order,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, eig_eiv_all, incidence_all):
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, eig_eiv_all, incidence_all)

        return x_all
#%%
# class SCCNN_Exp_2(nn.Module):

#     def __init__(self, in_channels: int, hidden_channels: int, n_layers: int) -> None:
#         super().__init__()

#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.n_layers = n_layers

#         # Stack multiple SCoNe layers with given hidden dimensions
#         self.layers = nn.ModuleList(
#             [SCCNN_Exp_Layer_2(self.in_channels, self.hidden_channels)]
#         )
#         for _ in range(self.n_layers - 1):
#             self.layers.append(SCCNN_Exp_Layer_2(self.hidden_channels, self.hidden_channels))

#         # Initialize parameters
#         for layer in self.layers:
#             layer.reset_parameters()

#     def forward(
#         self, x: torch.Tensor,
#         evals_d1: torch.Tensor, evecs_d1: torch.Tensor,
#         evals_u1: torch.Tensor, evecs_u1: torch.Tensor,
#     ) -> torch.Tensor:
#         """Forward pass through the network."""
#         for layer in self.layers:
#             x = layer(x, evals_d1, evecs_d1, evals_u1, evecs_u1)
#         return x