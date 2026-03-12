import torch 
import torch.nn as nn
import torch.nn.functional as F
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
        
        # nn.init.constant_(self.diffusion_time, 0.0)
        # nn.init.uniform_(self.diffusion_time, a=0.0, b=1.0)
        nn.init.uniform_(self.diffusion_time, a=0.0, b=50.0)
        
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.betta = nn.Parameter(torch.tensor(0.0))
    
    def reset_parameters(self):
        # self.Conv_layer.reset_parameters()            
        # nn.init.constant_(self.diffusion_time, 0.0)
        nn.init.uniform_(self.diffusion_time, a=0.0, b=50.0)
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
        time = self.diffusion_time
        
        # Same t for all channels
        if self.single_t:
            dim = x.shape[1]
            time = time.repeat(dim)

        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
        
        x_diffuse_spec = diffusion_coefs * x_spec

        # x_diffuse_spec = x_diffuse_spec -(self.alpha)*x_spec
        x_diffuse = torch.matmul(evecs, x_diffuse_spec)
        # x_diffuse = x_diffuse + (self.betta)*x
        # x_diffuse = self.Conv_layer(x_diffuse, L._indices(), edge_weight=L._values()).relu()  

        return x_diffuse

#%%
'''sccnn node, sccnn edge'''
class sccnn_conv_cont(nn.Module):
    def __init__(self, F_in, F_out, b1, b2, evals_0, evecs_0, evals_1l, evecs_1l, evals_1u, evecs_1u, evals_2, evecs_2, sigma, single_t):
        """
        F_in: number of input features per layer 
        F_out: number of output features per layer
        p: stands for positive, denoting the upper simplex order
        n: stands for negative, denoting the lower simplex order
        """
        super(sccnn_conv_cont, self).__init__()
        self.F_in = F_in
        self.F_out = F_out 
        self.B1 = b1 
        self.B2 = b2
        self.evals_0 = evals_0
        self.evecs_0 = evecs_0
        self.evals_1l = evals_1l
        self.evecs_1l = evecs_1l
        self.evals_1u = evals_1u
        self.evecs_1u = evecs_1u
        self.evals_2 = evals_2
        self.evecs_2 = evecs_2
        self.sigma = sigma
        self.single_t = single_t
        
        
        self.W0 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 4)))
        self.W1 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 7)))
        self.W2 = nn.parameter.Parameter(torch.empty(size=(self.F_in, self.F_out, 4)))
        
        dim_0 = self.evecs_0.size(dim=0)
        self.I0 = torch.eye(dim_0,device=self.evecs_0.device)
        dim_1 = self.evecs_1l.size(dim=0)
        self.I1 = torch.eye(dim_1,device=self.evecs_1l.device)
        dim_2 = self.evecs_2.size(dim=0)
        self.I2 = torch.eye(dim_2,device=self.evecs_2.device)
        
        
        self.diff_derivative_00 = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_0p = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_1nl = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_1l = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_1u = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_1pu = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_2n = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        self.diff_derivative_22 = Time_derivative_diffusion(single_t=self.single_t, C_inout=F_in)      
        
        self.reset_parameters()
        print("created SCCNN-Cont layers")
    
    def reset_parameters(self):
        """reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W0.data, gain=gain)
        nn.init.xavier_uniform_(self.W1.data, gain=gain)
        nn.init.xavier_uniform_(self.W2.data, gain=gain)
        
    
    def forward(self,x_in):
        x0,x1,x2 = x_in
        
        '''order 0 '''
        I0x = torch.unsqueeze(self.I0@x0,2)
        x0p = self.B1@x1
        I0xp = torch.unsqueeze(self.I0@x0p,2) # torch.inverse(self.D1)@self.B1 is the projection matrix
        
        X00 = torch.unsqueeze(self.diff_derivative_00(x0, self.evals_0, self.evecs_0), 2)   
        X0p = torch.unsqueeze(self.diff_derivative_0p(x0p, self.evals_0, self.evecs_0), 2)   
        X0 = torch.cat((I0x,X00,I0xp,X0p),2)

        '''order 1'''
        x1n = self.B1.T @ x0
        I1xn = torch.unsqueeze(self.I1@x1n,2)
        I1x = torch.unsqueeze(self.I1@x1,2)
        x1p = self.B2@x2
        I1xp = torch.unsqueeze(self.I1@x1p,2)
        
        X1nl = torch.unsqueeze(self.diff_derivative_1nl(x1n, self.evals_1l, self.evecs_1l), 2)   
        X1n = torch.cat((I1xn,X1nl),2)
        
        X1l = torch.unsqueeze(self.diff_derivative_1l(x1, self.evals_1l, self.evecs_1l), 2)   
        X1u = torch.unsqueeze(self.diff_derivative_1u(x1, self.evals_1u, self.evecs_1u), 2)   
        X11 = torch.cat((I1x, X1l, X1u),2)
            
        X1pu = torch.unsqueeze(self.diff_derivative_1pu(x1p, self.evals_1u, self.evecs_1u), 2)   
        X1p = torch.cat((I1xp, X1pu), 2)
            
        X1 = torch.cat((X1n,X11,X1p),2)
            
        '''order 2'''
        x2n = self.B2.T@x1
        I2xn = torch.unsqueeze(self.I2@x2n,2)
        I2x = torch.unsqueeze(self.I2@x2,2)
     
        X2n = torch.unsqueeze(self.diff_derivative_2n(x2n, self.evals_2, self.evecs_2), 2)   
        X22 = torch.unsqueeze(self.diff_derivative_22(x2, self.evals_2, self.evecs_2), 2)   
        X2 = torch.cat((I2xn, X2n, I2x, X22), 2)

        y0 = torch.einsum('nik,iok->no',X0,self.W0)
        y1 = torch.einsum('nik,iok->no',X1,self.W1)
        y2 = torch.einsum('nik,iok->no',X2,self.W2)
        y0 = self.sigma(y0)
        y1 = self.sigma(y1)
        y2 = self.sigma(y2)
        return y0,y1,y2