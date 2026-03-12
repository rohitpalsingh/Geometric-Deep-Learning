# -*- coding: utf-8 -*-
"""Results_Shrec16_Full.ipynb

# Train a COSIMO

### We train the model to perform:
    Complex Regression using the shrec16 benchmark dataset.

## Continuous Simplicial Neural Networks [COSIMO]</a>

# Complex Regression on the Shrec-16 Dataset: Full Version
"""

import os
print("Current working directory:", os.getcwd())
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working directory set to:", os.getcwd())

import numpy as np
import random
import torch
# One can define a fixed seed value (we spanned [1,10])
SEED = 5

# 1. Set the Python built-in random module's seed
random.seed(SEED)

# 2. Set the NumPy random seed
np.random.seed(SEED)

# 3. Set the PyTorch seed (for both CPU and GPU)
torch.manual_seed(SEED)# Define a fixed seed value
print(torch.cuda.is_available())

import toponetx.datasets as datasets
from sklearn.model_selection import train_test_split

from COSIMO import COSIMO
from topomodelx.utils.sparse import from_sparse

# %load_ext autoreload
# %autoreload 2

import scipy
from scipy import sparse

def get_evals_evecs(L, k):
    L_sparse = sparse.coo_matrix(L)

    evals, evecs = scipy.sparse.linalg.eigs(L_sparse, k=k, ncv=4*k, return_eigenvectors=True)
    # evals, evecs = scipy.linalg.eig(L)

    evals=torch.tensor(evals.real)
    evecs=torch.tensor(evecs.real)

    return evals, evecs

"""## Pre-processing

### Import shrec dataset ##

We must first lift our graph dataset into the simplicial complex domain.
"""

shrec, _ = datasets.mesh.shrec_16(size="full")
shrec = {key: np.array(value) for key, value in shrec.items()}
x_0s = shrec["node_feat"]
x_1s = shrec["edge_feat"]
x_2s = shrec["face_feat"]

ys = shrec["label"]
simplexes = shrec["complexes"]

in_channels_0 = x_0s[-1].shape[1]
in_channels_1 = x_1s[-1].shape[1]
in_channels_2 = x_2s[-1].shape[1]

in_channels_all = (in_channels_0, in_channels_1, in_channels_2)
print(in_channels_all)

"""### Define Neighborhood Strctures
Get incidence matrices $\mathbf{B}_1,\mathbf{B}_2$ and Hodge Laplacians $\mathbf{L}_0, \mathbf{L}_1$ and $\mathbf{L}_2$.
"""

max_rank = 2  # the order of the SC is two
incidence_1_list = []
incidence_2_list = []


kk_0 = 10
kk_1 = 10
kk_2 = 10
evals_0_list = []
evecs_0_list = []
evals_d1_list = []
evecs_d1_list = []
evals_u1_list = []
evecs_u1_list = []
evals_2_list = []
evecs_2_list = []

for simplex in simplexes:
    incidence_1 = simplex.incidence_matrix(rank=1)
    incidence_2 = simplex.incidence_matrix(rank=2)
    laplacian_0 = simplex.hodge_laplacian_matrix(rank=0)
    laplacian_down_1 = simplex.down_laplacian_matrix(rank=1)
    laplacian_up_1 = simplex.up_laplacian_matrix(rank=1)
    laplacian_2 = simplex.hodge_laplacian_matrix(rank=2)

    incidence_1 = from_sparse(incidence_1)
    incidence_2 = from_sparse(incidence_2)
    evals_0, evecs_0 = get_evals_evecs(laplacian_0, kk_0)
    evals_d1, evecs_d1 = get_evals_evecs(laplacian_down_1, kk_1)
    evals_u1, evecs_u1 = get_evals_evecs(laplacian_up_1, kk_1)
    evals_2, evecs_2 = get_evals_evecs(laplacian_2, kk_2)

    incidence_1_list.append(incidence_1)
    incidence_2_list.append(incidence_2)
    evals_0_list.append(evals_0)
    evecs_0_list.append(evecs_0)
    evals_d1_list.append(evals_d1)
    evecs_d1_list.append(evecs_d1)
    evals_u1_list.append(evals_u1)
    evecs_u1_list.append(evecs_u1)
    evals_2_list.append(evals_2)
    evecs_2_list.append(evecs_2)

print(laplacian_0.shape[0])
print(laplacian_down_1.shape[0])
print(laplacian_2.shape[0])

"""# Create and Train the Neural Network

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
        n_layers=2,
    ):
        super().__init__()
        self.base_model = COSIMO(
            in_channels_all=in_channels_all,
            hidden_channels_all=hidden_channels_all,
            conv_order=conv_order,
            sc_order=max_rank,
            n_layers=n_layers,
        )
        out_channels_0, out_channels_1, out_channels_2 = hidden_channels_all
        self.out_linear_0 = torch.nn.Linear(out_channels_0, out_channels)
        self.out_linear_1 = torch.nn.Linear(out_channels_1, out_channels)
        self.out_linear_2 = torch.nn.Linear(out_channels_2, out_channels)

    def forward(self, x_all, eig_eiv_all, incidence_all):
        x_all = self.base_model(x_all, eig_eiv_all, incidence_all)
        x_0, x_1, x_2 = x_all

        x_0 = self.out_linear_0(x_0)
        x_1 = self.out_linear_1(x_1)
        x_2 = self.out_linear_2(x_2)

        # Take the average of the 2D, 1D, and 0D cell features. If they are NaN, convert them to 0.
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        # Return the sum of the averages
        return (
            two_dimensional_cells_mean
            + one_dimensional_cells_mean
            + zero_dimensional_cells_mean
        )

conv_order = 2
intermediate_channels_all = (16, 16, 16)
num_layers = 2
out_channels = 1  # num classes

model = Network(
    in_channels_all=in_channels_all,
    hidden_channels_all=intermediate_channels_all,
    out_channels=out_channels,
    conv_order=conv_order,
    max_rank=max_rank,
    n_layers=num_layers,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss(size_average=True, reduction="mean")
print(model)

test_size = 0.2
val_size = 0.2
x_0_train, x_0_test = train_test_split(x_0s, test_size=test_size, shuffle=False)
x_0_train, x_0_val = train_test_split(x_0_train, test_size=val_size, shuffle=False)
x_1_train, x_1_test = train_test_split(x_1s, test_size=test_size, shuffle=False)
x_1_train, x_1_val = train_test_split(x_1_train, test_size=val_size, shuffle=False)
x_2_train, x_2_test = train_test_split(x_2s, test_size=test_size, shuffle=False)
x_2_train, x_2_val = train_test_split(x_2_train, test_size=val_size, shuffle=False)

incidence_1_train, incidence_1_test = train_test_split(
    incidence_1_list, test_size=test_size, shuffle=False
)
incidence_1_train, incidence_1_val = train_test_split(
    incidence_1_train, test_size=val_size, shuffle=False
)

incidence_2_train, incidence_2_test = train_test_split(
    incidence_2_list, test_size=test_size, shuffle=False
)
incidence_2_train, incidence_2_val = train_test_split(
    incidence_2_train, test_size=val_size, shuffle=False
)

evals_0_train, evals_0_test = train_test_split(
    evals_0_list, test_size=test_size, shuffle=False
)
evals_0_train, evals_0_val = train_test_split(
    evals_0_train, test_size=val_size, shuffle=False
)

evecs_0_train, evecs_0_test = train_test_split(
    evecs_0_list, test_size=test_size, shuffle=False
)
evecs_0_train, evecs_0_val = train_test_split(
    evecs_0_train, test_size=val_size, shuffle=False
)

evals_d1_train, evals_d1_test = train_test_split(
    evals_d1_list, test_size=test_size, shuffle=False
)
evals_d1_train, evals_d1_val = train_test_split(
    evals_d1_train, test_size=val_size, shuffle=False
)

evecs_d1_train, evecs_d1_test = train_test_split(
    evecs_d1_list, test_size=test_size, shuffle=False
)
evecs_d1_train, evecs_d1_val = train_test_split(
    evecs_d1_train, test_size=val_size, shuffle=False
)

evals_u1_train, evals_u1_test = train_test_split(
    evals_u1_list, test_size=test_size, shuffle=False
)
evals_u1_train, evals_u1_val = train_test_split(
    evals_u1_train, test_size=val_size, shuffle=False
)

evecs_u1_train, evecs_u1_test = train_test_split(
    evecs_u1_list, test_size=test_size, shuffle=False
)
evecs_u1_train, evecs_u1_val = train_test_split(
    evecs_u1_train, test_size=val_size, shuffle=False
)

evals_2_train, evals_2_test = train_test_split(
    evals_2_list, test_size=test_size, shuffle=False
)
evals_2_train, evals_2_val = train_test_split(
    evals_2_train, test_size=val_size, shuffle=False
)

evecs_2_train, evecs_2_test = train_test_split(
    evecs_2_list, test_size=test_size, shuffle=False
)
evecs_2_train, evecs_2_val = train_test_split(
    evecs_2_train, test_size=val_size, shuffle=False
)

y_train, y_test = train_test_split(ys, test_size=test_size, shuffle=False)
y_train, y_val = train_test_split(y_train, test_size=val_size, shuffle=False)

"""We train the COSIMO:"""

test_interval = 1
num_epochs = 100
Val_loss_Best = float('inf')

for epoch_i in range(1, num_epochs + 1):
    epoch_loss = []
    model.train()
    for (
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_2,
        evals_0, evecs_0,
        evals_d1, evecs_d1,
        evals_u1, evecs_u1,
        evals_2, evecs_2,
        y,
    ) in zip(
        x_0_train,
        x_1_train,
        x_2_train,
        incidence_1_train,
        incidence_2_train,
        evals_0_train, evecs_0_train,
        evals_d1_train, evecs_d1_train,
        evals_u1_train, evecs_u1_train,
        evals_2_train, evecs_2_train,
        y_train,
        strict=False,
    ):
        x_0 = torch.tensor(x_0)
        x_1 = torch.tensor(x_1)
        x_2 = torch.tensor(x_2)
        y = torch.tensor(y, dtype=torch.float)
        optimizer.zero_grad()
        x_all = (x_0.float(), x_1.float(), x_2.float())
        eig_eiv_all = (evals_0, evecs_0, evals_d1, evecs_d1, evals_u1, evecs_u1, evals_2, evecs_2)
        incidence_all = (incidence_1, incidence_2)

        y_hat = model(x_all, eig_eiv_all, incidence_all)

        # print(y_hat)
        loss = loss_fn(y_hat, y)

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    print(
        f"Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f}",
        flush=True,
    )
    with torch.no_grad():
            for (
                x_0,
                x_1,
                x_2,
                incidence_1,
                incidence_2,
                evals_0, evecs_0,
                evals_d1, evecs_d1,
                evals_u1, evecs_u1,
                evals_2, evecs_2,
                y,
            ) in zip(
                x_0_val,
                x_1_val,
                x_2_val,
                incidence_1_val,
                incidence_2_val,
                evals_0_val, evecs_0_val,
                evals_d1_val, evecs_d1_val,
                evals_u1_val, evecs_u1_val,
                evals_2_val, evecs_2_val,
                y_val,
                strict=False,
            ):
                x_0 = torch.tensor(x_0)
                x_1 = torch.tensor(x_1)
                x_2 = torch.tensor(x_2)
                y = torch.tensor(y, dtype=torch.float)
                optimizer.zero_grad()
                x_all = (x_0.float(), x_1.float(), x_2.float())
                eig_eiv_all = (
                    evals_0, evecs_0,
                    evals_d1, evecs_d1,
                    evals_u1, evecs_u1,
                    evals_2, evecs_2,
                )
                incidence_all = (incidence_1, incidence_2)

                y_hat = model(x_all, eig_eiv_all, incidence_all)

                Val_loss = loss_fn(y_hat, y)
            print(f"Val_loss: {loss:.4f}", flush=True)
            if Val_loss < Val_loss_Best:
                for (
                    x_0,
                    x_1,
                    x_2,
                    incidence_1,
                    incidence_2,
                    evals_0, evecs_0,
                    evals_d1, evecs_d1,
                    evals_u1, evecs_u1,
                    evals_2, evecs_2,
                    y,
                ) in zip(
                    x_0_test,
                    x_1_test,
                    x_2_test,
                    incidence_1_test,
                    incidence_2_test,
                    evals_0_test, evecs_0_test,
                    evals_d1_test, evecs_d1_test,
                    evals_u1_test, evecs_u1_test,
                    evals_2_test, evecs_2_test,
                    y_test,
                    strict=False,
                ):
                    x_0 = torch.tensor(x_0)
                    x_1 = torch.tensor(x_1)
                    x_2 = torch.tensor(x_2)
                    y = torch.tensor(y, dtype=torch.float)
                    optimizer.zero_grad()
                    x_all = (x_0.float(), x_1.float(), x_2.float())
                    eig_eiv_all = (
                        evals_0, evecs_0,
                        evals_d1, evecs_d1,
                        evals_u1, evecs_u1,
                        evals_2, evecs_2,
                    )
                    incidence_all = (incidence_1, incidence_2)

                    y_hat = model(x_all, eig_eiv_all, incidence_all)

                    Test_loss = loss_fn(y_hat, y)/(torch.norm(y,2)**2)
                print(f"Test_loss-improved: {Test_loss:.4f}", flush=True)
                Val_loss_Best = Val_loss
            else:
                print(f"Test_loss-still: {Test_loss:.4f}", flush=True)

            print(">"*100)