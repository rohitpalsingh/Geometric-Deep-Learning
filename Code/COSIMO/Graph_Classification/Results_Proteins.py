import os
print("Current working directory:", os.getcwd())
# print("File exists?", os.path.exists('./data/graph_classification/proteins/label_sets_proteins.npy'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working directory set to:", os.getcwd())

acc_test = []
for itter in range(10):
    Test_accuracy = -1
    import numpy as np
    import random
    import torch 
    # Define a fixed seed value
    SEED = 2

    # 1. Set the Python built-in random module's seed
    random.seed(SEED)

    # 2. Set the NumPy random seed
    np.random.seed(SEED)

    # 3. Set the PyTorch seed (for both CPU and GPU)
    torch.manual_seed(itter)# Define a fixed seed value
    print(torch.cuda.is_available())

    import toponetx.datasets as datasets
    from sklearn.model_selection import train_test_split

    from Utils.sccnn_exp import SCCNN
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

    import argparse

    import numpy as np
    from tqdm import tqdm
    import gudhi
    import torch
    from Utils.preprocessing.simplicial_construction import get_boundary_matrices, get_boundary_matrices_from_processed_tree, process_simplex_tree, get_neighbors, get_weight_matrix_graph, get_weight_matrix_simplex,generate_triangles, augment_simplex_open_gc, _get_laplacians,_get_simplex_features_gc
    from Utils.preprocessing.graph_construction import _get_graph
    # from model.model import MPSN,SCNN,SAN
    import torch.nn as nn
    # from model.loss import l_rel, l_sub
    import copy
    import time
    import numpy as np
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import torch
    import networkx as nx
    from sklearn import metrics
    from sklearn.metrics import classification_report,f1_score, accuracy_score
    import sys

    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    import gc
    gc.enable()



    parser = argparse.ArgumentParser(description='TopoSRL')

    parser.add_argument('--dataname', type=str, default='proteins', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of TopoSRL encoder.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay of TopoSRL encoder.')

    parser.add_argument('--dim', type=int, default=4, help='Order of the simplicial complex.')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha.')
    parser.add_argument('--snn', type=str, default='MPSN', help='Type of SNN')
    parser.add_argument('--delta', type=int, default=20, help='Number of samples to calculate L_rel')
    parser.add_argument('--augmentation', type=str,  default='open', help='Type of agumentation')
    parser.add_argument('--rho', type=float, default=0.1, help='Simplex removing and adding ratio.')

    args = parser.parse_args(args=[])

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'


    data = args.dataname
    alpha = args.alpha
    delta = args.delta
    epochs = args.epochs
    labels = np.load('./data/graph classification/'+data+'/label_sets_'+data+'.npy', allow_pickle=True)
    simplicial = np.load('./data/graph classification/'+data+'/simplicial_sets_'+data+'.npy', allow_pickle=True)
    SCs = []
    INDs = []
    _labels = []
    simplex_trees = []
    node_attributes = []
    netxG = []
    for p in range(len(simplicial)):
        for q in range(len(simplicial[p])):
            simplex_tree = gudhi.SimplexTree()
            sc = [[] for i in range(4)]
            for i in range(4):
                for j in simplicial[p][q][i]:
                    sc[i].append(list(j))
                    simplex_tree.insert(list(j))
            for i in range(len(sc)):
                sc[i] = np.array(sc[i])
            if(len(sc[3])):
                INDs.append(simplicial[p][q])
                g = nx.from_edgelist(sc[1])
                _labels.append(labels[p][q])
                node_attributes.append(nx.adjacency_matrix(g).todense().sum(1))
                netxG.append(g)
                SCs.append(sc)    
                simplex_trees.append(simplex_tree)
    labels = np.array(_labels)
    print("Length of dataset:", len(SCs))
    print(labels.shape)
    print(len(SCs))


    from scipy.sparse import coo_matrix


    max_rank = 2  # the order of the SC is two
    incidence_1_list = []
    incidence_2_list = []


    kk_0 = 2
    kk_1 = 2
    kk_2 = 2
    evals_0_list = []
    evecs_0_list = []
    evals_d1_list = []
    evecs_d1_list = []
    evals_u1_list = []
    evecs_u1_list = []
    evals_2_list = []
    evecs_2_list = []


    _labels = []
    skipped = []
    x_0s = []
    x_1s = []
    x_2s = []
    for i in tqdm(range(len(labels))):
        _,_,bm, = get_boundary_matrices_from_processed_tree(simplex_trees[i], SCs[i], INDs[i], 3)
        l1, l1_d, l1_u = _get_laplacians(bm)
        _X = torch.FloatTensor(node_attributes[i]).to(args.device).view(len(node_attributes[i]),1)
        X1 = _get_simplex_features_gc(SCs[i][1:3],_X)
        if (X1[0].shape[0] != l1[0].shape[0]) or (X1[1].shape[0] != l1[1].shape[0]) or (X1[2].shape[0] != l1[2].shape[0]):
            print(i)
            continue
        try:
            x_0s.append(X1[0].to(args.device))
            x_1s.append(X1[1].to(args.device))
            x_2s.append(X1[2].to(args.device))
            incidence_1 = bm[0].cpu().detach().numpy()
            incidence_2 = bm[1].cpu().detach().numpy()
            laplacian_0 = l1[0].cpu().detach().numpy()
            laplacian_down_1 = l1_d[1].cpu().detach().numpy()
            laplacian_up_1 = l1_u[1].cpu().detach().numpy()
            laplacian_2 = l1[2].cpu().detach().numpy()

            if X1[0].shape[0] != laplacian_0[0].shape[0]:
                print(i)
                print(X1[0].shape[0])
                print(laplacian_0[0].shape[0])

            incidence_1 = coo_matrix(incidence_1) # Convert NumPy array to COO sparse format
            incidence_2 = coo_matrix(incidence_2)  # Convert NumPy array to COO sparse format
            incidence_1 = from_sparse(incidence_1).to(args.device)
            incidence_2 = from_sparse(incidence_2).to(args.device)
            
            evals_0, evecs_0 = get_evals_evecs(laplacian_0, kk_0)
            evals_d1, evecs_d1 = get_evals_evecs(laplacian_down_1, kk_1)
            evals_u1, evecs_u1 = get_evals_evecs(laplacian_up_1, kk_1)
            evals_2, evecs_2 = get_evals_evecs(laplacian_2, kk_2)

            incidence_1_list.append(incidence_1)
            incidence_2_list.append(incidence_2)
            evals_0_list.append(evals_0.to(args.device))
            evecs_0_list.append(evecs_0.to(args.device))
            evals_d1_list.append(evals_d1.to(args.device))
            evecs_d1_list.append(evecs_d1.to(args.device))
            evals_u1_list.append(evals_u1.to(args.device))
            evecs_u1_list.append(evecs_u1.to(args.device))
            evals_2_list.append(evals_2.to(args.device))
            evecs_2_list.append(evecs_2.to(args.device))
            _labels.append(labels[i])
            # print(i)
            # print(laplacian_0)
        except RuntimeError:
            skipped.append(i)
    _labels = np.array(_labels)         


    print(laplacian_0.shape[0])
    print(laplacian_down_1.shape[0])
    print(laplacian_2.shape[0])
    print(len(x_0s))
    print(x_0s[1].shape)
    print(len(x_1s))
    print(x_1s[1].shape)
    print(len(x_2s))
    print(x_2s[1].shape)
    print(_labels.shape[0])

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
            self.base_model = SCCNN(
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
    out_channels = 2  # num classes

    # in_channels_0 = x_0s[-1].shape[1]
    # in_channels_1 = x_1s[-1].shape[1]
    # in_channels_2 = x_2s[-1].shape[1]

    in_channels_all = (1, 1, 1)
    # print(in_channels_all)

    model = Network(
        in_channels_all=in_channels_all,
        hidden_channels_all=intermediate_channels_all,
        out_channels=out_channels,
        conv_order=conv_order,
        max_rank=max_rank,
        n_layers=num_layers,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = torch.nn.MSELoss(size_average=True, reduction="mean")
    print(model)


    test_size = 0.2
    val_size = 0.2
    x_0_train, x_0_test = train_test_split(x_0s, test_size=test_size, shuffle=True, random_state=SEED)
    x_0_train, x_0_val = train_test_split(x_0_train, test_size=val_size, shuffle=True, random_state=SEED)
    x_1_train, x_1_test = train_test_split(x_1s, test_size=test_size, shuffle=True, random_state=SEED)
    x_1_train, x_1_val = train_test_split(x_1_train, test_size=val_size, shuffle=True, random_state=SEED)
    x_2_train, x_2_test = train_test_split(x_2s, test_size=test_size, shuffle=True, random_state=SEED)
    x_2_train, x_2_val = train_test_split(x_2_train, test_size=val_size, shuffle=True, random_state=SEED)

    incidence_1_train, incidence_1_test = train_test_split(
        incidence_1_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    incidence_1_train, incidence_1_val = train_test_split(
        incidence_1_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    incidence_2_train, incidence_2_test = train_test_split(
        incidence_2_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    incidence_2_train, incidence_2_val = train_test_split(
        incidence_2_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evals_0_train, evals_0_test = train_test_split(
        evals_0_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evals_0_train, evals_0_val = train_test_split(
        evals_0_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evecs_0_train, evecs_0_test = train_test_split(
        evecs_0_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evecs_0_train, evecs_0_val = train_test_split(
        evecs_0_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evals_d1_train, evals_d1_test = train_test_split(
        evals_d1_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evals_d1_train, evals_d1_val = train_test_split(
        evals_d1_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evecs_d1_train, evecs_d1_test = train_test_split(
        evecs_d1_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evecs_d1_train, evecs_d1_val = train_test_split(
        evecs_d1_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evals_u1_train, evals_u1_test = train_test_split(
        evals_u1_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evals_u1_train, evals_u1_val = train_test_split(
        evals_u1_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evecs_u1_train, evecs_u1_test = train_test_split(
        evecs_u1_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evecs_u1_train, evecs_u1_val = train_test_split(
        evecs_u1_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evals_2_train, evals_2_test = train_test_split(
        evals_2_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evals_2_train, evals_2_val = train_test_split(
        evals_2_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    evecs_2_train, evecs_2_test = train_test_split(
        evecs_2_list, test_size=test_size, shuffle=True, random_state=SEED
    )
    evecs_2_train, evecs_2_val = train_test_split(
        evecs_2_train, test_size=val_size, shuffle=True, random_state=SEED
    )

    # y_train, y_test = train_test_split(ys, test_size=test_size, shuffle=True)
    # y_train, y_val = train_test_split(y_train, test_size=val_size, shuffle=True)


    import torch.nn.functional as F

    y = np.array(_labels)
    print(y)
    num_classes = 2  # Define the number of classes
    one_hot_labels = np.array(F.one_hot(torch.tensor(y), num_classes=num_classes))


    # y_train = np.array(one_hot_labels[:320])
    # y_test = np.array(one_hot_labels[320:])

    y_train, y_test = train_test_split(one_hot_labels,test_size=test_size, shuffle=True, random_state=SEED)
    y_train, y_val = train_test_split(y_train, test_size=val_size, shuffle=True, random_state=SEED)
    # train_indices, test_indices = train_test_split(np.arange(len(one_hot_labels)), test_size=0.2, random_state=42)
    # y_train = one_hot_labels[train_indices]
    # y_test = one_hot_labels[test_indices]

    y_train = torch.from_numpy(y_train).to(args.device)
    y_test = torch.from_numpy(y_test).to(args.device)
    y_val = torch.from_numpy(y_val).to(args.device)

    test_interval = 1
    num_epochs = 30
    Val_loss_Best = float('inf')
    criterion = torch.nn.CrossEntropyLoss()
    Val_accuracy_Best = -1
    for epoch_i in range(1, num_epochs + 1):
        epoch_loss = []
        y_train_pred = []
        y_val_pred = []
        y_test_pred = []
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
            y_train_pred.append(y_hat)
            loss = criterion(y_hat, torch.argmax(y.float()))

            # print(y_hat)
            # loss = loss_fn(y_hat, y)

            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        probs = torch.softmax(torch.stack(y_train_pred, 0), dim=1)
        # Get predictions (index of the max probability)
        y_pred = torch.argmax(probs, dim=1)
        # accuracy = (y_pred[: len(y_train)] == torch.argmax(y_train.float(), dim=1)).float().mean().item()
        correct = (y_pred == torch.argmax(y_train.float(), dim=1)).sum().item()
        # correct = (y_pred[train_indices] == torch.argmax(y_train.float(), dim=1)).sum().item()
        accuracy = correct / y_train.size(0)

        print(
            f"iter: {itter} Epoch: {epoch_i} loss: {np.mean(epoch_loss):.4f} Train_acc: {accuracy:.2f}",
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

                    y_val_pred.append(y_hat)
                    # Val_loss = loss_fn(y_hat, y)
                    
                probs = torch.softmax(torch.stack(y_val_pred,0), dim=1)
                y_pred = torch.argmax(probs, dim=1)
                correct = (y_pred == torch.argmax(y_val.float(), dim=1)).sum().item()
                Val_accuracy = correct / y_val.size(0)

                    
                print(f"Val_acc: {Val_accuracy:.4f}", flush=True)
                if Val_accuracy > Val_accuracy_Best:
                    Val_accuracy_Best = Val_accuracy
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
                        y_test_pred.append(y_hat)

                    probs = torch.softmax(torch.stack(y_test_pred,0), dim=1)
                    y_pred = torch.argmax(probs, dim=1)
                    correct = (y_pred == torch.argmax(y_test.float(), dim=1)).sum().item()
                    Test_accuracy = correct / y_test.size(0)
                    # Test_loss = loss_fn(y_hat, y)/(torch.norm(y,2)**2)
                    print(f"Test_acc-improved: {Test_accuracy:.4f}", flush=True)
                else:
                    print(f"Test_acc-still: {Test_accuracy:.4f}", flush=True)
                    
                print(">"*100)
    acc_test.append(Test_accuracy)
