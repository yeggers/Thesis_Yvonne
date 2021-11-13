from typing import NamedTuple, Tuple

import numpy as np
import torch
from norse.torch.functional.lif import LIFFeedForwardState
from norse.torch.module.lif import LIFParameters
from norse.torch.module.lif_refrac import (LIFRefracCell,
                                           LIFRefracFeedForwardCell,
                                           LIFRefracParameters)

#import tools.importing_dvs
import tools.plotting as plot
from functional.lif_mod2 import (LIFModFeedForwardStateNT,
                                 LIFModFeedForwardStateWTA, get_max_pst)
from module.lif_mod import (LIFModFeedForwardCell, LIFModParameters,
                            LIFModParametersNeuronTrace)
from module.noDyn import (noDynFeedForwardCell, noDynFeedForwardState,
                          noDynParameters)
from SNN_param import SNN_param


class dim():
    """ This classs computes the output dimensions of the various layers based on the used strike, kernel and padding values.

    Parameters: 
        par (SNN_par): class defining network parameters
      
    """

    def __init__(
        self, par
    ):
        #Network parameters
        self.par= par 

        #Input layer
        self.conv_input_dim = self.get_after_conv_dim(self.par.height, self.par.width, self.par.input_kernel3D, self.par.input_stride3D, self.par.input_padding3D, self.par.input_m)

        #Merge layer
        self.conv_merge_dim = self.get_after_conv_dim(self.conv_input_dim[0], self.conv_input_dim[1], self.par.merge_kernel3D, self.par.merge_stride3D, self.par.merge_padding3D, self.par.merge_m)
        
        #MS-Conv layer
        self.conv_MSConv_dim = self.get_after_conv_dim(self.conv_merge_dim[0], self.conv_merge_dim[1], self.par.MSConv_kernel3D, self.par.MSConv_stride3D, self.par.MSConv_padding3D, self.par.MSConv_m)

        #Pooling layer
        self.conv_pooling_dim = self.get_after_conv_dim(self.conv_MSConv_dim[0], self.conv_MSConv_dim[1], self.par.pooling_kernel3D, self.par.pooling_stride3D, self.par.pooling_padding3D, self.par.pooling_m)

        #Dense layer (computing input dimension since it is a linear layer)
        self.fc_dense_dim = self.conv_pooling_dim

    def get_after_conv_dim(self, H, W, k, s, p, D = 1, d = (1,1,1)):
        '''Computing the dimensions of a neural layer after convolution has been applied

            Parameters: 
                H: Height of input image 
                W: Width of input image 
                k: Kernel
                s: Stride
                p: Padding 
                d: dilation 
                D: depth of input image (number of synapses between two neurons)
        '''
        
        H_out = np.floor((H + 2*p[1] - d[1] * (k[1] -1) - 1)/s[1] + 1)
        W_out = np.floor((W + 2*p[2] - d[2] * (k[2] -1) - 1)/s[2] + 1)
        D_out = np.floor((D + 2*p[0] - d[0] * (k[0] -1) - 1)/s[0] + 1)

        return int(H_out), int(W_out), int(D_out)
    

class Input():
    """ This classs defines the input layer of the SNN in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"

    Parameters: 
        par (SNN_par): class defining network parameters
        dt (float): simulation time step [s]
      
    """

    def __init__(
        self, par, dt
    ):
        #Network parameters
        self.par = par 

        #Time step 
        self.dt = dt

        #Compute output dimension of inut layer 
        self.conv_input_dim = dim(self.par).conv_input_dim
        
        #Setting up noDyn parameters 
        self.params_input = noDynParameters(alpha_mod = self.par.merge_alpha, lambda_x = self.par.merge_lambda_X, delays = self.par.merge_delay, m = self.par.merge_m)
        
        #Setting up neuron feedforward cell 
        self.noDyn_input = noDynFeedForwardCell((self.par.input_out_dim, self.conv_input_dim[0], self.conv_input_dim[1]), p = self.params_input, dt = self.dt)

    def forward(
        self, 
        x_ts : torch.tensor, 
        s_input : noDynFeedForwardState,
        batch_size : int,
        downsample_parameters : tuple
    )-> Tuple[torch.Tensor, noDynFeedForwardState]:

        """ Compute output of input layer  

        Parameters: 
            x_ts (torch.Tensor): input sequence at current times step
            s_input (noDynFeedforwardState): state of the input layer
            batch_size (int): number of sequences to be processed at same time
            downsample_parameters (tuple): parameters for downsampling of input data (kernel_size, stride, padding)    
        """

        #Downsampling data
        x_ts = torch.nn.MaxPool2d(*downsample_parameters)(x_ts)

        #Computing output spikes and neuron states 
        z, s_input = self.noDyn_input(batch_size, x_ts, s_input)

        return z, s_input


class Merge():
    """ This classs defines the Merge layer of the SNN in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"

    Parameters: 
        par (SNN_par): class defining network parameters
        dt (float): simulation time step [s]
      
    """

    def __init__(
        self, par, dt
    ):
        #Network parameters
        self.par = par 

        #Time step 
        self.dt = dt

        #Compute output dimension of Merge layer 
        self.conv_merge_dim = dim(self.par).conv_input_dim
        
        #Setting up LIF parameters 
        self.params_merge = LIFModParameters(LIFRefracParameters(LIFParameters(tau_syn_inv = 1/self.par.merge_lambda_i, tau_mem_inv = 1/self.par.merge_lambda_v, v_th = self.par.merge_v_th), rho_reset = self.par.delta_ref), delays = torch.linspace(self.par.MsConv_tau_min, self.par.MsConv_tau_max, self.par.MSConv_m).to(dtype=torch.long), m =self.par.MSConv_m)

        #Setting up modified LIF parameters 
        self.modParams_merge = LIFModParametersNeuronTrace(lifMod = self.params_merge,  alpha_mod = self.par.MSConv_alpha, lambda_x = self.par.MSConv_lambda_X)
        
        #Setting up neuron feedforward cell 
        self.lif_merge = LIFModFeedForwardCell(
            (self.par.merge_out_dim,self.conv_merge_dim[0],  self.conv_merge_dim[1]),
            p= self.modParams_merge, dt = self.dt
        )

    def forward(
        self, 
        z : torch.tensor, 
        s_merge : LIFModFeedForwardStateNT,
        batch_size : int,
        device: torch.device,
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:

        """ Compute output of merge layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_merge (LIFModFeedforwardStateNT): state of the merge layer
            batch_size (int): number of sequences to be processed at same time
            device (torch.devide): device on which computations are performed  
        """

        #Convolving input spikes with weights 
        input_tensor= torch.nn.functional.conv3d(z, self.par.merge_weights.to(device), None, self.par.merge_stride3D, self.par.merge_padding3D)
       
        #Computing output spikes and neuron states
        z, s_merge = self.lif_merge.forward_NT(batch_size, input_tensor, s_merge, torch.tensor(self.par.merge_kernel3D).to(device), torch.tensor(self.par.merge_stride3D).to(device), device)

        return z, s_merge



class MSConv():
    """ This classs defines the MSConv layer of the SNN in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"

    Parameters: 
        par (SNN_par): class defining network parameters
        dt (float): simulation time step [s]
      
    """

    def __init__(
        self, par, dt
    ):
        #Network parameters
        self.par = par 

        #Time step 
        self.dt = dt

        #Compute output dimension of MSConv layer 
        self.conv_MSConv_dim = dim(self.par).conv_MSConv_dim
        
        #Setting up LIF parameters 
        self.params_MSConv = LIFRefracParameters(LIFParameters(tau_syn_inv = 1/self.par.MSConv_lambda_i, tau_mem_inv = 1/self.par.MSConv_lambda_v, v_th = self.par.MSConv_v_th), rho_reset = self.par.delta_ref)
        
        #Setting up modified LIF parameters 
        self.modParams_MSConv = LIFModParameters(lifRefrac = self.params_MSConv, delays = self.par.pooling_delay, m = self.par.pooling_m)
        
        #Setting up neuron feedforward cell 
        self.lif_MSConv = LIFModFeedForwardCell(
            (self.par.MSConv_out_dim,self.conv_MSConv_dim[0],  self.conv_MSConv_dim[1]),
            p=self.modParams_MSConv, dt = self.dt
        )

    def forward(
        self, 
        z : torch.tensor, 
        s_MSConv : LIFModFeedForwardStateWTA,
        batch_size : int,
        s_STDP_weights_exc : torch.tensor,
        s_STDP_weights_inh : torch.tensor,
        X: torch.tensor,
        device: torch.device,
        training: bool
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:

        """ Compute output of MSConv layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_MSConv (LIFModFeedforwardStateWTA): state of the MSConv layer
            batch_size (int): number of sequences to be processed at same time
            s_STDP_weights_exc (torch.tensor): current excitatory weights of the MSConv layer  
            s_STDP_weights_inh (torch.tensor): current excitatory weights of the MSConv layer  
            X (torch.tensor): presynaptic trace (neuron trace of previous layer)
            device (torch.devide): device on which computations are performed
            training (bool): indicates whether or not layer is being trained  
        """

        #Convolving input spikes with weights for excitatory and inhibitory weigts
        input_tensor_ex = torch.nn.functional.conv3d(z, s_STDP_weights_exc.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        input_tensor_inh = torch.nn.functional.conv3d(z, s_STDP_weights_inh.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        
        #Combining exitatory and inhibitory weights 
        input_tensor = input_tensor_ex + self.par.MSConv_beta*input_tensor_inh
        
        #Convolving presynaptic traces for each neuron 
        X_tensor = torch.nn.functional.conv3d(X, self.par.MSConv_weights_pst.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        
        #Determine maximum convolved spikerain within neighborhood of each neuron 
        X_tensorm = torch.nn.MaxPool2d(3, 1, 1)(X_tensor.squeeze(2))
        
        #Computing output spikes and neuron states
        z, s_MSConv, boxes, box_indices = self.lif_MSConv.forward_WTA(batch_size, input_tensor - X_tensorm.unsqueeze(2), s_MSConv, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device), device, training = training)

        return z, s_MSConv, boxes, box_indices, input_tensor - X_tensorm.unsqueeze(2)




