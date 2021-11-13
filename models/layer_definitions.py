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
                                 LIFModFeedForwardStateWTA, compute_tau_max_update)
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

        #SS-Conv layer
        self.conv_SSConv_dim = self.get_after_conv_dim(self.conv_input_dim[0], self.conv_input_dim[1], self.par.SSConv_kernel3D, self.par.SSConv_stride3D, self.par.SSConv_padding3D, self.par.SSConv_m)

        #Merge layer
        self.conv_merge_dim = self.get_after_conv_dim(self.conv_SSConv_dim[0], self.conv_SSConv_dim[1], self.par.merge_kernel3D, self.par.merge_stride3D, self.par.pooling_padding3D, self.par.merge_m)

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
        self.params_input = noDynParameters(alpha_mod = self.par.SSConv_alpha, lambda_x = self.par.SSConv_lambda_X, delays = self.par.SSConv_delay, m = self.par.SSConv_m)
        
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


class SSConv():
    """ This classs defines the SSConv layer of the SNN in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception"

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

        #Compute output dimension of SSConv layer 
        self.conv_SSConv_dim = dim(self.par).conv_SSConv_dim
        
        #Setting up LIF parameters 
        self.params_SSConv = LIFRefracParameters(LIFParameters( tau_mem_inv = 1/self.par.SSConv_lambda_v, v_th = self.par.SSConv_v_th), rho_reset = self.par.delta_ref)
        
        #Setting up modified LIF parameters 
        self.modParams_SSConv = LIFModParameters(lifRefrac = self.params_SSConv, delays = self.par.merge_delay, m = self.par.merge_m, lambda_vth = self.par.SSConv_lambda_vth, alpha_vth = self.par.SSConv_alpha_vth, vth_rest = self.par.SSConv_vth_rest, vth_conv_params = self.par.SSConv_vth_conv_params, stiffness_goal_inf = self.par.SSConv_stiffness_goal_inf, target_parameters = self.par.SSConv_target_parameters, len_stiffness_buffer = self.par.SSConv_len_stiffness_buffer, v_th_gains = self.par.SSConv_v_th_gains)
        
        #Setting up neuron feedforward cell 
        self.lif_SSConv = LIFModFeedForwardCell(
            (self.par.SSConv_out_dim, self.conv_SSConv_dim[0],  self.conv_SSConv_dim[1]), p= self.modParams_SSConv, dt = self.dt
        )

    def forward(
        self, 
        z : torch.tensor, 
        s_SSConv : LIFModFeedForwardStateWTA,
        batch_size : int,
        s_STDP_weights : torch.tensor,
        X: torch.tensor,
        device: torch.device,
        training: bool,
        t: int
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:

        """ Compute output of SSConv layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_SSConv (LIFModFeedforwardStateWTA): state of the SSConv layer
            batch_size (int): number of sequences to be processed at same time
            s_STDP_weights (torch.tensor): current weights of the SSConv layer  
            X (torch.tensor): presynaptic trace (neuron trace of previous layer)
            device (torch.devide): device on which computations are performed
            training (bool): indicates whether or not layer is being trained  
        """

        #Convolving input spikes with weights 
        input_tensor = torch.nn.functional.conv3d(z, s_STDP_weights.to(device), None, self.par.SSConv_stride3D, self.par.SSConv_padding3D)

        #Convolving presynaptic traces for each neuron 
        X_tensor = torch.nn.functional.conv3d(X, self.par.SSConv_weights_pst.to(device), None, self.par.SSConv_stride3D, self.par.SSConv_padding3D)
   
        #Determine maximum convolved spiketrain within neighborhood of each neuron 
        X_tensorm, max_indices = torch.nn.MaxPool2d(5, 1, 2, return_indices = True)(X_tensor.squeeze(2))
  
        #Computing output spikes and neuron states: Max PSTs
        z, s_SSConv, v_th, boxes, box_indices, v_decayed, stiffness_ave, stiffness_goal = self.lif_SSConv.forward_WTA(X_tensor, batch_size, input_tensor - X_tensorm.unsqueeze(2), s_SSConv, torch.tensor(self.par.SSConv_kernel3D).to(device), torch.tensor(self.par.SSConv_stride3D).to(device), device, t, training = training)

        # #Computing output spikes and neuron states: individual PSTs
        # z, s_SSConv, boxl, boxu, boxes, box_indices, spike_indices_maps, spike_indices_batch, num_spikes, winds_maps, winds_batch, v_decayed = self.lif_SSConv.forward_WTA(X_tensor, batch_size, input_tensor, s_SSConv, torch.tensor(self.par.SSConv_kernel3D).to(device), torch.tensor(self.par.SSConv_stride3D).to(device), device, training = training)

        return z, s_SSConv, v_th, boxes, box_indices, v_decayed, max_indices, stiffness_ave, stiffness_goal



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

        #Compute output dimension of SSConv layer 
        self.conv_merge_dim = dim(self.par).conv_SSConv_dim
        
        #Setting up LIF parameters 
        self.params_merge = LIFModParameters(LIFRefracParameters(LIFParameters(tau_mem_inv = 1/self.par.merge_lambda_v, v_th = self.par.merge_v_th), rho_reset = self.par.delta_ref), delays =self.par.MSConv_delay, m =self.par.MSConv_m)

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

        #Compute output dimension of SSConv layer 
        self.conv_MSConv_dim = dim(self.par).conv_MSConv_dim
        
        #Setting up LIF parameters 
        self.params_MSConv = LIFRefracParameters(LIFParameters(tau_mem_inv = 1/self.par.MSConv_lambda_v, v_th = self.par.MSConv_v_th), rho_reset = self.par.delta_ref)
        
        #Setting up modified LIF parameters 
        self.modParams_MSConv = LIFModParameters(lifRefrac = self.params_MSConv, delays = self.par.pooling_delay, m = self.par.pooling_m, lambda_vth = self.par.MSConv_lambda_vth, alpha_vth = self.par.MSConv_alpha_vth, vth_rest = self.par.MSConv_vth_rest, vth_conv_params = self.par.MSConv_vth_conv_params, stiffness_goal_inf = self.par.MSConv_stiffness_goal_inf, target_parameters = self.par.MSConv_target_parameters, len_stiffness_buffer = self.par.MSConv_len_stiffness_buffer, v_th_gains = self.par.MSConv_v_th_gains, tau_max_gain = self.par.MSConv_tau_max_gain)
        
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
        training: bool, 
        t: int, 
        STD_goal: torch.tensor
    )-> Tuple[torch.Tensor, LIFModFeedForwardStateWTA]:

        """ Compute output of SSConv layer  

        Parameters: 
            z (torch.Tensor): input spikes 
            s_MSConv (LIFModFeedforwardStateWTA): state of the SSConv layer
            batch_size (int): number of sequences to be processed at same time
            s_STDP_weights_exc (torch.tensor): current excitatory weights of the SSConv layer  
            s_STDP_weights_inh (torch.tensor): current excitatory weights of the SSConv layer  
            X (torch.tensor): presynaptic trace (neuron trace of previous layer)
            device (torch.devide): device on which computations are performed
            training (bool): indicates whether or not layer is being trained  
            t (int): current time step
            STD_goal: target value for tau STD
        """

        #Convolving input spikes with weights for excitatory and inhibitory weigts
        input_tensor_ex = torch.nn.functional.conv3d(z, s_STDP_weights_exc.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        input_tensor_inh = torch.nn.functional.conv3d(z, s_STDP_weights_inh.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        
        #Combining exitatory and inhibitory weights 
        input_tensor = input_tensor_ex + self.par.MSConv_beta*input_tensor_inh
        
        #Convolving presynaptic traces for each neuron 
        X_tensor = torch.nn.functional.conv3d(X, self.par.MSConv_weights_pst.to(device), None, self.par.MSConv_stride3D, self.par.MSConv_padding3D)
        
        X_tensorm, max_indices = torch.nn.MaxPool2d(7, 1, 3, return_indices = True)(X_tensor.squeeze(2))
        
        #Computing output spikes and neuron states
        z, s_MSConv, v_th, boxes, box_indices, v_decayed, stiffness_ave, stiffness_goal = self.lif_MSConv.forward_WTA(X_tensor, batch_size, input_tensor - X_tensorm.unsqueeze(2), s_MSConv, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device), device, t,  training = training)

        #TODO: think of a better more consistent place to put this
        #Update maximum delay 
        tau_max = self.par.MsConv_tau_max
        # if training == False:
        #     tau_max = compute_tau_max_update(s_STDP_weights_exc, s_STDP_weights_inh, torch.where(z>0)[1], STD_goal, self.par)

        return z, s_MSConv, v_th, tau_max, boxes, box_indices, v_decayed, max_indices, stiffness_ave, stiffness_goal




