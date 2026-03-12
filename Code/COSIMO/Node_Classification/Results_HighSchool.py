# -*- coding: utf-8 -*-
"""Results_HighSchool.ipynb
"""

# !pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html

import os
print("Current working directory:", os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working directory set to:", os.getcwd())

"""# Fixing some settings:"""

import numpy as np
import random
import torch
# Define a fixed seed value
SEED = 2 # We selected the SEEDs in [1,10]
test_size = 0.2
val_size = 0.2
num_filters = 16 # Number of features for learnable filters

# 1. Set the Python built-in random module's seed
random.seed(SEED)

# 2. Set the NumPy random seed
np.random.seed(SEED)

# 3. Set the PyTorch seed (for both CPU and GPU)
torch.manual_seed(SEED)# Define a fixed seed value
print(torch.cuda.is_available())

import toponetx.datasets as datasets
from sklearn.model_selection import train_test_split

from Utils.sccnn_exp import COSIMO
from topomodelx.utils.sparse import from_sparse

# %load_ext autoreload
# %autoreload 2

"""A Function for performing EVD on Laplacians and selecting the first $k$ eigenvalue-eigenvector pairs:"""

import scipy
from scipy import sparse

def get_evals_evecs(L, k):
    L_sparse = sparse.coo_matrix(L)

    evals, evecs = scipy.sparse.linalg.eigs(L_sparse, k=k, ncv=4*k, return_eigenvectors=True)
    # evals, evecs = scipy.linalg.eig(L)

    evals=torch.tensor(evals.real)
    evecs=torch.tensor(evecs.real)

    return evals, evecs

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

from Utils.preprocessing.simplicial_construction import get_boundary_matrices, get_neighbors, get_weight_matrix_graph, get_weight_matrix_simplex,generate_triangles,_get_laplacians,_get_simplex_features,augment_simplex,augment_simplex_open
import argparse
from Utils.preprocessing.graph_construction import _get_graph
import torch
import networkx as nx

parser = argparse.ArgumentParser(description='TopoSRL')

parser.add_argument('--dataname', type=str, default='contact-high-school', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--dim', type=int, default=4, help='Order of the simplicial complex.')

args = parser.parse_args(args=[])

if args.gpu != -1 and torch.cuda.is_available():
    # args.device = 'cuda:{}'.format(args.gpu)
    args.device = 'cuda'
else:
    args.device = 'cpu'
print(args.device)

simplex_tree, sc, boundry_matrices, labels =  get_boundary_matrices(args.dataname, args.dim)
print("Got boundaries")
g, netxG = _get_graph(sc[1])
g = g.to(args.device)
A = nx.adjacency_matrix(netxG).todense()
sm = torch.nn.Softmax(dim=1)
W0 = get_weight_matrix_graph(A)
W0 = sm(torch.FloatTensor(W0).to(args.device))
W0 = W0 * (W0!=W0.min(axis=1).values.unsqueeze(-1))
laplacians, lower_laplacians, upper_laplacians = _get_laplacians(boundry_matrices)
_X = _get_simplex_features(sc[1:4], g.ndata['features'])
print(labels)

"""### Define Neighborhood Strctures
Get incidence matrices $\mathbf{B}_1,\mathbf{B}_2$ and Hodge Laplacians $\mathbf{L}_0, \mathbf{L}_1$ and $\mathbf{L}_2$.
"""

incidence_1 = boundry_matrices[0].cpu().detach().numpy()
incidence_2 = boundry_matrices[1].cpu().detach().numpy()

print(f"The incidence matrix B1 has shape: {incidence_1.shape}.")
print(f"The incidence matrix B2 has shape: {incidence_2.shape}.")

laplacian_0 = laplacians[0]
laplacian_down_1 = lower_laplacians[1]
laplacian_up_1 = upper_laplacians[1]
laplacian_down_2 = lower_laplacians[2]
laplacian_up_2 = upper_laplacians[2]

print(laplacian_0.shape)
print(laplacian_down_1.shape)
print(laplacian_up_1.shape)
print(laplacian_down_2.shape)
print(laplacian_up_2.shape)

kk = 10 # Number of selected eigenvalue-eigenvector pairs:
evals_0, evecs_0 = get_evals_evecs(laplacian_0.cpu().detach(), kk)
evals_d1, evecs_d1 = get_evals_evecs(laplacian_down_1.cpu().detach(), kk)
evals_u1, evecs_u1 = get_evals_evecs(laplacian_up_1.cpu().detach(), kk)
evals_d2, evecs_d2 = get_evals_evecs(laplacian_down_2.cpu().detach(), kk)
evals_u2, evecs_u2 = get_evals_evecs(laplacian_up_2.cpu().detach(), kk)

from scipy.sparse import coo_matrix
incidence_1 = coo_matrix(incidence_1)  # Convert NumPy array to COO sparse format
incidence_2 = coo_matrix(incidence_2)  # Convert NumPy array to COO sparse format

incidence_1 = from_sparse(incidence_1)
incidence_2 = from_sparse(incidence_2)

"""## Import signals ##"""

"""A function to obtain features based on the input: rank
"""
def get_simplicial_features(dataset, rank):
    if rank == 0:
        which_feat = "node_feat"
    elif rank == 1:
        which_feat = "edge_feat"
    elif rank == 2:
        which_feat = "face_feat"
    else:
        raise ValueError(
            "input dimension must be 0, 1 or 2, because features are supported on nodes, edges and faces"
        )

    x = list(dataset.get_simplex_attributes(which_feat).values())
    return torch.tensor(np.stack(x))

x_0 = _X[0]
x_1 = _X[1]
x_2 = _X[2]
print(f"There are {x_0.shape[0]} nodes with features of dimension {x_0.shape[1]}.")
print(f"There are {x_1.shape[0]} edges with features of dimension {x_1.shape[1]}.")
print(f"There are {x_2.shape[0]} faces with features of dimension {x_2.shape[1]}.")

"""## Define binary labels
We retrieve the labels associated to the nodes of each input simplex. In the KarateClub dataset, two social groups emerge. So we assign binary labels to the nodes indicating of which group they are a part.

We convert the binary labels into one-hot encoder form, and keep the first four nodes' true labels for the purpose of testing.
"""

import torch.nn.functional as F

y = np.array(labels-1)
print(y)
num_classes = 9  # Define the number of classes
one_hot_labels = np.array(F.one_hot(torch.tensor(y), num_classes=num_classes))

y_train, y_test = train_test_split(one_hot_labels,test_size=test_size, shuffle=False)
y_train = torch.from_numpy(y_train).to(args.device)
y_test = torch.from_numpy(y_test).to(args.device)

"""# Create and Train the Continuous Simplcial Neural Network (COSIMO)

We specify the model with our pre-made neighborhood structures and specify an optimizer.
"""

class Network(torch.nn.Module):
    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        out_channels,
        conv_order,
        max_rank,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        self.base_model = COSIMO(
            in_channels_all=in_channels_all,
            hidden_channels_all=hidden_channels_all,
            conv_order=conv_order,
            sc_order=max_rank,
            update_func=update_func,
            n_layers=n_layers,
        )
        out_channels_0, _, _ = hidden_channels_all
        self.out_linear_0 = torch.nn.Linear(out_channels_0, out_channels)

    def forward(self, x_all, eig_eiv_all, incidence_all):
        x_all_1 = self.base_model(x_all, eig_eiv_all, incidence_all)
        x_0, _, _ = x_all_1

        logits = self.out_linear_0(x_0)
        return logits

"""Obtain the initial features on all simplices"""
x_all = (x_0.to(args.device), x_1.to(args.device), x_2.to(args.device))

# Defining some settings and hyperparameters:
conv_order = 2
in_channels_all = (x_0.shape[-1], x_1.shape[-1], x_2.shape[-1])
intermediate_channels_all = (num_filters, num_filters, num_filters)
num_layers = 4
out_channels = num_classes  # num classes
max_rank = 4

eig_eiv_all = (
    evals_0.to(args.device), evecs_0.to(args.device),
    evals_d1.to(args.device), evecs_d1.to(args.device),
    evals_u1.to(args.device), evecs_u1.to(args.device),
    evals_d2.to(args.device), evecs_d2.to(args.device),
    evals_u2.to(args.device), evecs_u2.to(args.device),
)

incidence_all = (incidence_1.to(args.device), incidence_2.to(args.device))

model = Network(
    in_channels_all=in_channels_all,
    hidden_channels_all=intermediate_channels_all,
    out_channels=out_channels,
    conv_order=conv_order,
    max_rank=max_rank,
    update_func=None,
    n_layers=num_layers,
)

model = model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

test_interval = 10
num_epochs = 700
val_accuracy_best = -1
test_accuracy = -1
# Define cross-entropy loss function
criterion = torch.nn.CrossEntropyLoss()
for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    optimizer.zero_grad()
    y_hat = model(x_all, eig_eiv_all, incidence_all)

    # Compute loss
    loss = criterion(y_hat[: len(y_train)], torch.argmax(y_train.float(), dim=1))

    epoch_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    probs = torch.softmax(y_hat, dim=1)
    # Get predictions (index of the max probability)
    y_pred = torch.argmax(probs, dim=1)
    correct = (y_pred[: len(y_train)] == torch.argmax(y_train.float(), dim=1)).sum().item()
    accuracy = correct / y_train.size(0)
    print(
        f"Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f} Train_acc: {accuracy:.2f}",
        flush=True,
    )

    if epoch_i % test_interval == 0:
        with torch.no_grad():
            y_hat_test = model(x_all, eig_eiv_all, incidence_all)
            # Prodiction to node-level:
            probs = torch.softmax(y_hat_test, dim=1)
            # Get predictions (index of the max probability)
            y_pred_test = torch.argmax(probs, dim=1)
            correct = (y_pred_test[-len(y_test) :] == torch.argmax(y_test.float(), dim=1)).sum().item()
            test_accuracy = correct / y_test.size(0)

            print()
            print()
            print(f"Test_acc: {test_accuracy:.2f}", flush=True)
            print()
            print()

print(f"Test_acc: {test_accuracy:.2f}", flush=True)