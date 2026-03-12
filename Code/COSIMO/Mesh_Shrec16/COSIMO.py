"""COSIMO implementation for complex classification."""
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
from COSIMO_layer import COSIMO_Layer_exp


class COSIMO(torch.nn.Module):
    """COSIMO implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all: tuple of int
        Dimension of input features on (nodes, edges, faces).
    hidden_channels_all: tuple of int
        Dimension of features of hidden layers on (nodes, edges, faces).
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order: int
        Order of simplicial complex.
    aggr_norm: bool
        Whether to normalize the aggregation.
    update_func: str
        Update function for the simplicial complex convolution.
    n_layers: int
        Number of layers.

    """

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
            COSIMO_Layer_exp(
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

        # Forward through COSIMO
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, eig_eiv_all, incidence_all)

        return x_all