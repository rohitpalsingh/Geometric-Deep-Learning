"""Continuous Simplicial Neural Networks (COSIMO) Layer."""
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

import torch

from topomodelx.base.aggregation import Aggregation

#%%
class Time_derivative_diffusion(nn.Module):
    def __init__(self, C_inout, single_t):
        super(Time_derivative_diffusion, self).__init__()
        self.C_inout = C_inout
        self.single_t = single_t
        
        # same t for all channels
        if self.single_t:
            self.diffusion_time = nn.Parameter(torch.Tensor(1))
        else:
        # learnable t
            self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  

       
        # self.Conv_layer = GCN_diff(GCN_opt, C_inout, C_inout)

        
        # self.method = method # one of ['spectral', 'implicit_dense']
        # self.num_nodes = num_nodes 
        
        # nn.init.constant_(self.diffusion_time, 100)
        # nn.init.xavier_normal_(self.diffusion_time)
        # nn.init.uniform_(self.diffusion_time, a=0.0, b=1.0)
        # nn.init.xavier_uniform_(self.diffusion_time)
        nn.init.normal_(self.diffusion_time)
        
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        # self.Conv_layer.reset_parameters()            
        # nn.init.constant_(self.diffusion_time, 100)
        # nn.init.xavier_normal_(self.diffusion_time)
        # nn.init.uniform_(self.diffusion_time, a=0, b=30)
        nn.init.normal_(self.diffusion_time)
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.betta = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x, evals, evecs):
         

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))
        
            
        # Transform to spectral
        x_spec = torch.matmul(torch.transpose(evecs,1,0),x)
        
        # Diffuse
        time = torch.relu(self.diffusion_time)
        
        # Same t for all channels
        if self.single_t:
            dim = x.shape[-1]
            time = time.repeat(dim)

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
        
        x_diffuse_spec = diffusion_coefs * x_spec

        # x_diffuse_spec = x_diffuse_spec -(self.alpha)*x_spec
        x_diffuse = torch.matmul(evecs, x_diffuse_spec)
        # x_diffuse = x_diffuse + (self.betta)*x
        # x_diffuse = self.Conv_layer(x_diffuse, L._indices(), edge_weight=L._values()).relu()  

        return x_diffuse

#%%
class COSIMO_Layer(torch.nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features on nodes, edges, and triangles.
    out_channels : tuple of int
        Dimensions of output features on nodes, edges, and triangles.
    conv_order : int
        Convolution order of the simplicial filters.
        To avoid too many parameters, we consider them to be the same.
    sc_order : int
        SC order.
    aggr_norm : bool, default = False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : str, default = None
        Activation function used in aggregation layers.
    initialization : str, default = "xavier_normal"
        Weight initialization method.

    Examples
    --------
    Here we provide an example of pseudocode for COSIMO layer in an SC
    of order two
    input X_0: [n_nodes, in_channels]
    input X_1: [n_edges, in_channels]
    input X_2: [n_faces, in_channels]

    graph Laplacian L_0: [n_nodes, n_nodes]
    1-Lap_down L_1_down: [n_edges, n_edges]
    1-Lap_up L_1_up: [n_edges, n_edges]
    2-Lap L_2: [n_faces,n_faces]
    1-incidence B_1: [n_nodes, n_edges]
    2-incidence B_2: [n_edges, n_faces]

    conv_order: int, e.g., 2

    output Y_0: [n_nodes, out_channels]
    output Y_1: [n_edges, out_channels]
    output Y_2: [n_faces, out_channels]

    COSIMO layer looks like:

        Y_0 = torch.einsum(
        concat(
            X_0, L_0@X_0, L_0@L_0@X_0 ||
            B_1@X_1, B_1@L_1_down@X_1, B_1@L_1_down@L_1_down@X_1
        ), weight_0)
        Y_1 = torch.einsum(
        concat(
            B_1.T@X_1, B_1.T@L_0@X_0, B_1.T@L_0@L_0@X_0 ||
            X_1, L_1_down@X_1, L_1_down@L_1_down@X_1,
                L_1_up@X_1, L_1_up@L_1_up@X_1 ||
            B_2@X_2, B_2@L_2@X_2, B_2@L_2@L_2@X_2
        ), weight_1)
        Y_2 = torch.einsum(
        concat(
            X_2, L_2@X_2, L_2@L_2@X_2 ||
            B_2.T@X_1, B_2.T@L_1_up@X_1, B_2.T@L_1_up@L_1_up@X_1
        ), weight_2)
    where
        - weight_0, weight_2, weight_2 are the trainable parameters
        - weight_0: [out_channels, in_channels, total_order_0]
            - total_order_0 = 1+conv_order + 1+conv_order
        - weight_1: [out_channels, in_channels, total_order_1]
            - total_order_1 = 1+conv_order +
                              1+conv_order+conv_order +
                              1+conv_order
        - weight_2: [out_channels, in_channels, total_order_2]
            - total_order_2 = 1+conv_order + 1+conv_order
        - to implement Lap_down@Lap_down@X, we consider chebyshev method
            to avoid matrix@matrix computation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels_0 = out_channels_0
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_0 = Parameter(
            torch.Tensor(
                self.in_channels_0, self.out_channels_0, 1 + conv_order + 1 + conv_order
            )
        )

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                1 + conv_order + 1 + conv_order + conv_order + 1 + conv_order,
            )
        )

        # determine the third dimensions of the weights
        # because when SC order is larger than 2, there are lower and upper
        # parts for L_2; otherwise, L_2 contains only the lower part
        if sc_order > 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    1 + conv_order + 1 + conv_order + conv_order,
                )
            )
        elif sc_order == 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    1 + conv_order + 1 + conv_order,
                )
            )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization."""
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor, shape = (n_target_cells, out_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse, shape = (n_simplices,n_simplices)
            Convolution operator e.g., the adjacency matrix, or the Hodge Laplacians.
        conv_order : int
            The order of the convolution.
        x : torch.Tensor, shape = (n_simplices,num_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor. x[:, :, k] = (conv_operator@....@conv_operator) @ x.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order))

        if self.aggr_norm:
            X[:, :, 0] = torch.mm(conv_operator, x)
            X[:, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, 0])
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        else:
            X[:, :, 0] = torch.mm(conv_operator, x)
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
        return X

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation (see [1]_).

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0\rightarrow1)}  = B_1^T \cdot h_y^{t,(0)} \cdot \Theta^{t,(0 \rightarrow 1)}\\
            &🟧 $\quad m_{z}^{(0\rightarrow1)}  = \frac{1}\sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0\rightarrow1)} \qquad \text{where} \sum \text{represents a mean.}\\
            &🟥 $\quad m_{z \rightarrow x}^{(1 \rightarrow 0)} = B_1\odot att(m_{z \in \mathcal{C}(x)}^{(0\rightarrow1)}, h_x^{t,(0)}) \cdot m_z^{(0\rightarrow1)} \cdot \Theta^{t,(1 \rightarrow 0)}\\
            &🟧 $\quad m_x^{(1\rightarrow0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)} \qquad \text{where} \sum \text{represents a mean.}\\
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1, (0)} = \Theta^{t, \text{update}} \cdot (h_x^{t,(0)}||m_x^{(0)})+b^{t, \text{update}}\\
            \end{align*}

        Parameters
        ----------
        x_all : tuple of tensors, shape = (x_0,x_1,x_2)
            Tuple of input feature tensors:

            - x_0: torch.Tensor, shape = (n_nodes,in_channels_0),
            - x_1: torch.Tensor, shape = (n_edges,in_channels_1),
            - x_2: torch.Tensor, shape = (n_triangles,in_channels_2).

        laplacian_all: tuple of tensors, shape = (laplacian_0,laplacian_down_1,laplacian_up_1,laplacian_2)
            Tuple of laplacian tensors:

            - laplacian_0: torch.sparse, graph Laplacian,
            - laplacian_down_1: torch.sparse, the 1-Hodge laplacian (lower part),
            - laplacian_up_1: torch.sparse, the 1-hodge laplacian (upper part),
            - laplacian_2: torch.sparse, the 2-hodge laplacian.

        incidence_all : tuple of tensors, shape = (b1,b2)
            Tuple of incidence tensors:

            - b1: torch.sparse, shape = (n_nodes,n_edges), node-to-edge incidence matrix,
            - b2: torch.sparse, shape = (n_edges,n_triangles), edge-to-face incidence matrix.

        Returns
        -------
        y_0 : torch.Tensor
            Output features on nodes.
        y_1 : torch.Tensor
            Output features on edges.
        y_2 : torch.Tensor
            Output features on triangles.
        """
        x_0, x_1, x_2 = x_all

        if self.sc_order == 2:
            laplacian_0, laplacian_down_1, laplacian_up_1, laplacian_2 = laplacian_all
        elif self.sc_order > 2:
            (
                laplacian_0,
                laplacian_down_1,
                laplacian_up_1,
                laplacian_down_2,
                laplacian_up_2,
            ) = laplacian_all

        num_nodes, num_edges, num_triangles = x_0.shape[0], x_1.shape[0], x_2.shape[0]

        b1, b2 = incidence_all

        identity_0, identity_1, identity_2 = (
            torch.eye(num_nodes),
            torch.eye(num_edges),
            torch.eye(num_triangles),
        )

        """
        convolution in the node space
        """
        x_identity_0 = torch.unsqueeze(identity_0 @ x_0, 2)
        x_0_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        x_0_to_0 = torch.cat((x_identity_0, x_0_to_0), 2)

        x_1_to_0 = torch.mm(b1, x_1)
        x_1_to_0_identity = torch.unsqueeze(identity_0 @ x_1_to_0, 2)
        x_1_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_1_to_0)
        x_1_to_0 = torch.cat((x_1_to_0_identity, x_1_to_0), 2)

        x_0_all = torch.cat((x_0_to_0, x_1_to_0), 2)

        """
        convolution in the edge space
        """
        x_identity_1 = torch.unsqueeze(identity_1 @ x_1, 2)
        x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_1)
        x_1_to_1 = torch.cat((x_identity_1, x_1_down, x_1_up), 2)

        x_0_to_1 = torch.mm(b1.T, x_0)
        x_0_to_1_identity = torch.unsqueeze(identity_1 @ x_0_to_1, 2)
        x_0_to_1 = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_0_to_1)
        x_0_to_1 = torch.cat((x_0_to_1_identity, x_0_to_1), 2)

        x_2_to_1 = torch.mm(b2, x_2)
        x_2_to_1_identity = torch.unsqueeze(identity_1 @ x_2_to_1, 2)
        x_2_to_1 = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_2_to_1)
        x_2_to_1 = torch.cat((x_2_to_1_identity, x_2_to_1), 2)

        x_1_all = torch.cat((x_0_to_1, x_1_to_1, x_2_to_1), 2)

        """
        convolution in the face (triangle) space, depending on the SC order,
        the exact form maybe a little different
        """
        x_identity_2 = torch.unsqueeze(identity_2 @ x_2, 2)

        if self.sc_order == 2:
            x_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_2)
            x_2_to_2 = torch.cat((x_identity_2, x_2), 2)
        elif self.sc_order > 2:
            x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
            x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
            x_2_to_2 = torch.cat((x_identity_2, x_2_down, x_2_up), 2)

        x_1_to_2 = torch.mm(b2.T, x_1)
        x_1_to_2_identity = torch.unsqueeze(identity_2 @ x_1_to_2, 2)
        if self.sc_order == 2:
            x_1_to_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_1_to_2)
        elif self.sc_order > 2:
            x_1_to_2 = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_1_to_2)

        x_1_to_2 = torch.cat((x_1_to_2_identity, x_1_to_2), 2)

        x_2_all = torch.cat((x_2_to_2, x_1_to_2), 2)

        y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        if self.update_func is None:
            return y_0, y_1, y_2

        return self.update(y_0), self.update(y_1), self.update(y_2)
#%%
class COSIMO_Layer_exp(torch.nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features on nodes, edges, and triangles.
    out_channels : tuple of int
        Dimensions of output features on nodes, edges, and triangles.
    conv_order : int
        Convolution order of the simplicial filters.
        To avoid too many parameters, we consider them to be the same.
    sc_order : int
        SC order.
    aggr_norm : bool, default = False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : str, default = None
        Activation function used in aggregation layers.
    initialization : str, default = "xavier_normal"
        Weight initialization method.

    Examples
    --------
    Here we provide an example of pseudocode for COSIMO layer in an SC
    of order two
    input X_0: [n_nodes, in_channels]
    input X_1: [n_edges, in_channels]
    input X_2: [n_faces, in_channels]

    graph Laplacian L_0: [n_nodes, n_nodes]
    1-Lap_down L_1_down: [n_edges, n_edges]
    1-Lap_up L_1_up: [n_edges, n_edges]
    2-Lap L_2: [n_faces,n_faces]
    1-incidence B_1: [n_nodes, n_edges]
    2-incidence B_2: [n_edges, n_faces]

    conv_order: int, e.g., 2

    output Y_0: [n_nodes, out_channels]
    output Y_1: [n_edges, out_channels]
    output Y_2: [n_faces, out_channels]

    COSIMO layer looks like:

        Y_0 = torch.einsum(
        concat(
            X_0, L_0@X_0, L_0@L_0@X_0 ||
            B_1@X_1, B_1@L_1_down@X_1, B_1@L_1_down@L_1_down@X_1
        ), weight_0)
        Y_1 = torch.einsum(
        concat(
            B_1.T@X_1, B_1.T@L_0@X_0, B_1.T@L_0@L_0@X_0 ||
            X_1, L_1_down@X_1, L_1_down@L_1_down@X_1,
                L_1_up@X_1, L_1_up@L_1_up@X_1 ||
            B_2@X_2, B_2@L_2@X_2, B_2@L_2@L_2@X_2
        ), weight_1)
        Y_2 = torch.einsum(
        concat(
            X_2, L_2@X_2, L_2@L_2@X_2 ||
            B_2.T@X_1, B_2.T@L_1_up@X_1, B_2.T@L_1_up@L_1_up@X_1
        ), weight_2)
    where
        - weight_0, weight_2, weight_2 are the trainable parameters
        - weight_0: [out_channels, in_channels, total_order_0]
            - total_order_0 = 1+conv_order + 1+conv_order
        - weight_1: [out_channels, in_channels, total_order_1]
            - total_order_1 = 1+conv_order +
                              1+conv_order+conv_order +
                              1+conv_order
        - weight_2: [out_channels, in_channels, total_order_2]
            - total_order_2 = 1+conv_order + 1+conv_order
        - to implement Lap_down@Lap_down@X, we consider chebyshev method
            to avoid matrix@matrix computation

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
        single_t = True,
    ) -> None:
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels_0 = out_channels_0
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        self.single_t = single_t
        self.diff_derivative_00 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[0])      
        self.diff_derivative_10 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[0])      
        self.diff_derivative_d1 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[1])   
        self.diff_derivative_u1 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[1])   
        self.diff_derivative_01 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[1])   
        self.diff_derivative_21 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[1])   
        self.diff_derivative_2 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[2])   
        self.diff_derivative_d2 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[2])   
        self.diff_derivative_u2 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[2])   
        self.diff_derivative_12 = Time_derivative_diffusion(single_t=self.single_t, C_inout=out_channels[2])   


        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_0 = Parameter(
            torch.Tensor(
                self.in_channels_0, self.out_channels_0, 4
            )
        )

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                7,
            )
        )

        # determine the third dimensions of the weights
        # because when SC order is larger than 2, there are lower and upper
        # parts for L_2; otherwise, L_2 contains only the lower part
        if self.sc_order > 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    5,
                )
            )
        elif self.sc_order == 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4,
                )
            )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization."""
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor, shape = (n_target_cells, out_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        return None


    def forward(self, x_all, eig_eiv, incidence_all):
        r"""Forward computation (see [1]_).

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0\rightarrow1)}  = B_1^T \cdot h_y^{t,(0)} \cdot \Theta^{t,(0 \rightarrow 1)}\\
            &🟧 $\quad m_{z}^{(0\rightarrow1)}  = \frac{1}\sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0\rightarrow1)} \qquad \text{where} \sum \text{represents a mean.}\\
            &🟥 $\quad m_{z \rightarrow x}^{(1 \rightarrow 0)} = B_1\odot att(m_{z \in \mathcal{C}(x)}^{(0\rightarrow1)}, h_x^{t,(0)}) \cdot m_z^{(0\rightarrow1)} \cdot \Theta^{t,(1 \rightarrow 0)}\\
            &🟧 $\quad m_x^{(1\rightarrow0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)} \qquad \text{where} \sum \text{represents a mean.}\\
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1, (0)} = \Theta^{t, \text{update}} \cdot (h_x^{t,(0)}||m_x^{(0)})+b^{t, \text{update}}\\
            \end{align*}

        Parameters
        ----------
        x_all : tuple of tensors, shape = (x_0,x_1,x_2)
            Tuple of input feature tensors:

            - x_0: torch.Tensor, shape = (n_nodes,in_channels_0),
            - x_1: torch.Tensor, shape = (n_edges,in_channels_1),
            - x_2: torch.Tensor, shape = (n_triangles,in_channels_2).

        laplacian_all: tuple of tensors, shape = (laplacian_0,laplacian_down_1,laplacian_up_1,laplacian_2)
            Tuple of laplacian tensors:

            - laplacian_0: torch.sparse, graph Laplacian,
            - laplacian_down_1: torch.sparse, the 1-Hodge laplacian (lower part),
            - laplacian_up_1: torch.sparse, the 1-hodge laplacian (upper part),
            - laplacian_2: torch.sparse, the 2-hodge laplacian.

        incidence_all : tuple of tensors, shape = (b1,b2)
            Tuple of incidence tensors:

            - b1: torch.sparse, shape = (n_nodes,n_edges), node-to-edge incidence matrix,
            - b2: torch.sparse, shape = (n_edges,n_triangles), edge-to-face incidence matrix.

        Returns
        -------
        y_0 : torch.Tensor
            Output features on nodes.
        y_1 : torch.Tensor
            Output features on edges.
        y_2 : torch.Tensor
            Output features on triangles.
        """
        x_0, x_1, x_2 = x_all

        if self.sc_order == 2:
            evals_0, evecs_0, evals_d1, evecs_d1, evals_u1, evecs_u1, evals_2, evecs_2 = eig_eiv
        elif self.sc_order > 2:
            (
                evals_0, evecs_0,
                evals_d1, evecs_d1,
                evals_u1, evecs_u1,
                evals_d2, evecs_d2,
                evals_u2, evecs_u2,
            ) = eig_eiv

        num_nodes, num_edges, num_triangles = x_0.shape[0], x_1.shape[0], x_2.shape[0]

        b1, b2 = incidence_all

        identity_0, identity_1, identity_2 = (
            torch.eye(num_nodes),
            torch.eye(num_edges),
            torch.eye(num_triangles),
        )

        """
        convolution in the node space
        """
        x_identity_0 = torch.unsqueeze(identity_0 @ x_0, 2)
        # x_0_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        x_0_to_0 = torch.unsqueeze(self.diff_derivative_00(x_0, evals_0, evecs_0), 2)   
        x_0_to_0 = torch.cat((x_identity_0, x_0_to_0), 2)

        x_1_to_0 = torch.mm(b1, x_1)
        x_1_to_0_identity = torch.unsqueeze(identity_0 @ x_1_to_0, 2)
        x_1_to_0 = torch.unsqueeze(self.diff_derivative_10(x_1_to_0, evals_0, evecs_0), 2)   
        # x_1_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_1_to_0)
        x_1_to_0 = torch.cat((x_1_to_0_identity, x_1_to_0), 2)

        x_0_all = torch.cat((x_0_to_0, x_1_to_0), 2)

        """
        convolution in the edge space
        """
        x_identity_1 = torch.unsqueeze(identity_1 @ x_1, 2)
        # x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_down = torch.unsqueeze(self.diff_derivative_d1(x_1, evals_d1, evecs_d1), 2)   
        # x_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_1)
        x_1_up = torch.unsqueeze(self.diff_derivative_u1(x_1, evals_u1, evecs_u1), 2)   
        x_1_to_1 = torch.cat((x_identity_1, x_1_down, x_1_up), 2)

        x_0_to_1 = torch.mm(b1.T, x_0)
        x_0_to_1_identity = torch.unsqueeze(identity_1 @ x_0_to_1, 2)
        # x_0_to_1 = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_0_to_1)
        x_0_to_1 = torch.unsqueeze(self.diff_derivative_01(x_0_to_1, evals_d1, evecs_d1), 2)   
        x_0_to_1 = torch.cat((x_0_to_1_identity, x_0_to_1), 2)

        x_2_to_1 = torch.mm(b2, x_2)
        x_2_to_1_identity = torch.unsqueeze(identity_1 @ x_2_to_1, 2)
        # x_2_to_1 = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_2_to_1)
        x_2_to_1 = torch.unsqueeze(self.diff_derivative_21(x_2_to_1, evals_u1, evecs_u1), 2)   
        x_2_to_1 = torch.cat((x_2_to_1_identity, x_2_to_1), 2)

        x_1_all = torch.cat((x_0_to_1, x_1_to_1, x_2_to_1), 2)

        """
        convolution in the face (triangle) space, depending on the SC order,
        the exact form maybe a little different
        """
        x_identity_2 = torch.unsqueeze(identity_2 @ x_2, 2)

        if self.sc_order == 2:
            # x_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_2)
            x_2 = torch.unsqueeze(self.diff_derivative_2(x_2, evals_2, evecs_2), 2)   
            x_2_to_2 = torch.cat((x_identity_2, x_2), 2)
        elif self.sc_order > 2:
            # x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
            x_2_down = torch.unsqueeze(self.diff_derivative_d2(x_2, evals_d2, evecs_d2), 2)   
            # x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
            x_2_up = torch.unsqueeze(self.diff_derivative_u2(x_2, evals_u2, evecs_u2), 2)   
            x_2_to_2 = torch.cat((x_identity_2, x_2_down, x_2_up), 2)

        x_1_to_2 = torch.mm(b2.T, x_1)
        x_1_to_2_identity = torch.unsqueeze(identity_2 @ x_1_to_2, 2)
        if self.sc_order == 2:
            # x_1_to_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_1_to_2)
            x_1_to_2 = torch.unsqueeze(self.diff_derivative_12(x_1_to_2, evals_2, evecs_2), 2)   
        elif self.sc_order > 2:
            # x_1_to_2 = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_1_to_2)
            x_1_to_2 = torch.unsqueeze(self.diff_derivative_12(x_1_to_2, evals_d2, evecs_d2), 2)   

        x_1_to_2 = torch.cat((x_1_to_2_identity, x_1_to_2), 2)

        x_2_all = torch.cat((x_2_to_2, x_1_to_2), 2)

        y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        # print(x_2_all.shape)
        # print(self.weight_2.shape)
        # print(self.sc_order)
        y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        if self.update_func is None:
            return y_0, y_1, y_2

        return self.update(y_0), self.update(y_1), self.update(y_2)
#%%
"""Simplicial Complex Net Layer."""
class COSIMO_Exp_Layer_2(torch.nn.Module):
    """Implementation of the SCoNe layer proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Input dimension of features on each edge.
    out_channels : int
        Output dimension of features on each edge.
    update_func : Literal['relu', 'sigmoid', 'tanh']
        Update function to use when updating edge features.

    Notes
    -----
    This is the architecture proposed for trajectory prediction on simplicial complexes.

    For the trajectory prediction architecture proposed in [1]_, these layers are stacked before applying the
    boundary map from 1-chains to 0-chains. Finally, one can apply the softmax operator on the neighbouring nodes of
    the last node in the given trajectory to predict the next node. When implemented like this, we get a map from
    (ordered) 1-chains (trajectories) to the neighbouring nodes of the last node in the 1-chain.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        update_func: Literal["relu", "sigmoid", "tanh"] = "tanh",
        single_t=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_0 = torch.nn.parameter.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.weight_0_aux = torch.nn.parameter.Parameter(
            torch.Tensor(self.out_channels, self.out_channels)
        )
        self.weight_1 = torch.nn.parameter.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.weight_1_aux = torch.nn.parameter.Parameter(
            torch.Tensor(self.out_channels, self.out_channels)
        )
        self.weight_2 = torch.nn.parameter.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.weight_2_aux = torch.nn.parameter.Parameter(
            torch.Tensor(self.out_channels, self.out_channels)
        )
        self.aggr_on_edges = Aggregation("sum", update_func)
        Aggregation()

        self.single_t = single_t
        self.diff_derivative_d1 = Time_derivative_diffusion(single_t=self.single_t, C_inout=in_channels)   
        self.diff_derivative_u1 = Time_derivative_diffusion(single_t=self.single_t, C_inout=in_channels)   

    def reset_parameters(self, gain: float = 10.0) -> None:
        """Reset learnable parameters."""
        torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_0_aux, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_1_aux, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_2_aux, gain=gain)

    def forward(
        self, x: torch.Tensor,
        evals_d1: torch.Tensor, evecs_d1: torch.Tensor,
        evals_u1: torch.Tensor, evecs_u1: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass.

        The forward pass was initially proposed in [1]_.
        Its equations are given in [2]_ and graphically illustrated in [3]_.

        .. math::
            \begin{align*}
            &🟥 \quad m^{(1 \rightarrow 0 \rightarrow 1)}_{y \rightarrow \{z\} \rightarrow x}  = (L_{\downarrow,1})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟥 \quad m_{x \rightarrow x}^{(1 \rightarrow 1)}  = h_x^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 1)}\\
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}  = (L_{\uparrow,1})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 2 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1 \rightarrow 0 \rightarrow 1)} = \sum_{y \in \mathcal{L}_\downarrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1 \rightarrow 2 \rightarrow 1)}  = \sum_{y \in \mathcal{L}_\uparrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}\\
            &🟩 \quad m_x^{(1)}  = m_{x}^{(1 \rightarrow 0 \rightarrow 1)} + m_{x \rightarrow x}^{(1 \rightarrow 1)} + m_{x}^{(1 \rightarrow 2 \rightarrow 1)}\\
            &🟦 \quad h_x^{t,(1)} = \sigma(m_x^{(1)})
            \end{align*}

        Parameters
        ----------
        x: torch.Tensor, shape = (n_edges, in_channels)
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse, shape = (n_nodes, n_edges)
            Incidence matrix :math:`B_1` mapping edges to nodes.
        incidence_2 : torch.sparse, shape = (n_edges, n_triangles)
            Incidence matrix :math:`B_2` mapping triangles to edges.

        Returns
        -------
        torch.Tensor, shape = (n_edges, out_channels)
            Output features on the edges of the simplicial complex.
        """
        leaky_relu = nn.LeakyReLU()
        selu = nn.SELU()
        
        # z1 = incidence_2 @ incidence_2.T @ x @ self.weight_2
        # z1 = torch.relu(self.diff_derivative_u1(x, evals_u1, evecs_u1)@self.weight_2) @ self.weight_2_aux   
        z1 = torch.relu(self.diff_derivative_u1(x, evals_u1, evecs_u1)@self.weight_2) @ self.weight_2_aux   
        # z2 = torch.relu(x @ self.weight_1)@ self.weight_1_aux
        z2 = x @ self.weight_1
        # z3 = incidence_1.T @ incidence_1 @ x @ self.weight_0
        z3 = torch.relu(self.diff_derivative_d1(x, evals_d1, evecs_d1)@self.weight_0) @ self.weight_0_aux   
        return self.aggr_on_edges([z1, z2, z3])
