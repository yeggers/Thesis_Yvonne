
import os
import random
import time
from typing import NamedTuple, Tuple
import bz2
import pickle
import _pickle as cPickle
import cv2
import matplotlib.pyplot as plt
import norse
import numpy as np
import pylab
import torch
from matplotlib.animation import FuncAnimation

#import tools.importing_dvs
#import models.SNN_Frederico
#rom SNN_param import SNN_param
from tools.OF_vectors import compute_OF


import tools.plotting as plot

from module.lif_mod import LIFModFeedForwardCell, LIFModParameters

from module.noDyn import noDynFeedForwardCell, noDynParameters
from module.STDP_federico import (STDPFedericoFeedForwardCell,
                                    STDPFedericoParameters)
from functional.STDP_federico2 import STDPFedericoState

from models.layer_definitions import Input, SSConv, Merge, MSConv


class TrainMSConv(torch.nn.Module):
    """ This class can be used to train the MSConv layer in Federicos SNN.

    Parameters: 
        device (torch.device): device on which training should be performed (CPU/GPU)
        par (SNN_param): parameters of the SNN
        height (int): number of pixels in input images in vertical direction 
        width (int): number of pixels in input image in horizontal direction 
        method (string): 
        alpha (int): 
        dt(float): time step of the simulation 
    """
    
    def __init__(
        self, device, par, weights_name_exc, weights_name_inh, height = 180, width = 240, dt = 10**(-3), SSConvWeights = "SSConvWeights.pt", MSConvWeights_exc = "MSConvWeights_exc.pt", MSConvWeights_inh = "MSConvWeights_inh.pt"
    ):
        super(TrainMSConv, self).__init__()

   
        self.dt = dt
        self.par = par

        #Loading weights 
        self.SSConv_weights = torch.load(SSConvWeights).to(device).to(torch.float32)
        self.MSConv_weights_exc = torch.load(MSConvWeights_exc).to(device).to(torch.float32)
        self.MSConv_weights_inh = torch.load(MSConvWeights_inh).to(device).to(torch.float32)
        self.weights_name_exc = weights_name_exc
        self.weights_name_inh = weights_name_inh

        #Compute STD over delays for all maps for adaptive tau_max
        weights = self.MSConv_weights_exc + self.par.MSConv_beta * self.MSConv_weights_inh
        self.delay_weights = torch.sum(weights , dim = (1,3,4))
        self.STD_goal = torch.mean(torch.std(self.delay_weights, dim = 1)/torch.mean(self.delay_weights, dim = 1))
        #STD_weights.append(torch.mean(delay_weights_STD)) 

        #Initializing network layers 
        self.input = Input(self.par, self.dt)
        self.noDyn_input = self.input.noDyn_input

        self.SSConv = SSConv(self.par, self.dt)
        self.lif_SSConv = self.SSConv.lif_SSConv

        self.merge = Merge(self.par, self.dt)
        self.lif_merge = self.merge.lif_merge

        self.MSConv = MSConv(self.par, self.dt)
        self.lif_MSConv = self.MSConv.lif_MSConv

    
        #Setting up STDP rule 
        self.STDP_params_exc = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.MSConv_w_init_ex, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP_exc = STDPFedericoFeedForwardCell(self.par.MSConv_weight_shape, (self.par.MSConv_out_dim, self.MSConv.conv_MSConv_dim[0], self.MSConv.conv_MSConv_dim[1]), p = self.STDP_params_exc, dt = self.dt)

        self.STDP_params_inh = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.MSConv_w_init_inh, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP_inh = STDPFedericoFeedForwardCell(self.par.MSConv_weight_shape, (self.par.MSConv_out_dim, self.MSConv.conv_MSConv_dim[0], self.MSConv.conv_MSConv_dim[1]), p = self.STDP_params_inh, dt = self.dt)


    def forward(self,  
        dir : str,
        batch_size:int,
        iterations: int,
        training: bool,
        device: torch.device 
    ) -> Tuple[torch.Tensor, STDPFedericoState]:

        """ Compute STDP updates for SSConv layer 

        Parameters: 
            dir (str) : directory containing input sequences if random_file = True
            batch_size (int): number of sequences to be processed at same time
            iterations (int): number of iterations to use during training
            device (torch.device): device on which computations shall be performed
        """
      
        #Specifying initial state for STDP rule 
        s_STDP_exc = self.STDP_exc.initial_state(batch_size, self.MSConv_weights_exc, device = device, dtype = torch.float32 )
        s_STDP_inh = self.STDP_inh.initial_state(batch_size, self.MSConv_weights_inh, device = device, dtype = torch.float32 )

        #Initializing input layer 
        s_input = self.noDyn_input.initial_state(batch_size, device = device, dtype = torch.float32)
        #Initializing SSConv layer 
        s_SSConv = self.lif_SSConv.initial_state_WTA(batch_size, self.par.SSConv_len_stiffness_buffer, device = device, dtype = torch.float32)
        #Initializing Merge layer 
        s_merge = self.lif_merge.initial_state_NT(batch_size, device = device, dtype = torch.float32)
        #Initializing MSConv layer 
        s_MSConv = self.lif_MSConv.initial_state_WTA(batch_size, self.par.MSConv_len_stiffness_buffer, device = device, dtype = torch.float32)
        
        
        #Initialize containers to collect results
        stiffness_ave_buffer = []
        threshold_buffer = []
        stiffness_goal_buffer = []
        
    
        for sequence in range(iterations):
            print("Sequence: ", sequence)
            
            #Randomly selecting sequence from directory 
            random_files = random.sample(os.listdir(dir), batch_size)
            
            # #Loading first sequence
            # x = torch.load(dir + '/{}'.format(random_files[0]))

            data = bz2.BZ2File(dir + '/{}'.format(random_files[0]), 'rb')
            x = cPickle.load(data)

            #Determine sequence length
            seq_length = x.shape[0]

            #Adding remaining sequences in batch 
            for sequence in range(1,len(random_files)):
                file_name = dir + '/{}'.format(random_files[sequence])
                load = torch.load(file_name)
                x = torch.cat((x, load),dim = 1)
    
            #Randomly decide whether or not to flip image in x and y direction and polarity
            flip_w, flip_h, flip_p = tuple(np.random.randint(0,2,3))
            
            #Apllying flips
            x = (1 - flip_w) * x + flip_w * x[:,:, :,:,  torch.arange(x.shape[4] -1, -1, -1)]
            x = (1 - flip_h) * x + flip_h * x[:, :, :, torch.arange(x.shape[3] -1, -1, -1)]
            x = (1 - flip_p) * x + flip_p * x[:,:, torch.arange(x.shape[2] -1, -1, -1)]

            # #Plotting weight distribution 
            # plot.plot_histogram(s_STDP_exc.weights + self.par.MSConv_beta*s_STDP_inh.weights, 500, -1, 1, 3, 'Weights_exc [-]', device)
           
            
            #Initializing buffers for plotting 
            input_buffer = torch.zeros(10, batch_size, self.par.input_out_dim, self.par.SSConv_m, *self.input.conv_input_dim[0:2]).to(device)
            SSConv_buffer = torch.zeros(50, batch_size, self.par.SSConv_out_dim, self.par.merge_m, *self.SSConv.conv_SSConv_dim[0:2]).to(device)
            merge_buffer = torch.zeros(50, batch_size, self.par.merge_out_dim, self.par.MSConv_m, *self.merge.conv_merge_dim[0:2]).to(device)
            MSConv_buffer = torch.zeros(100, batch_size, self.par.MSConv_out_dim, self.par.pooling_m, *self.MSConv.conv_MSConv_dim[0:2]).to(device)

        
            for ts in range(seq_length):
                #Only putting input at current time step on GPU
                x_ts = x[ts, :].to(device)  

                #Compute total time
                time = sequence*seq_length + ts

                #Input layer
                # #Plotting input before downsampling
                # plot.plot2D_discrete(x_ts, self.par.height, self.par.width, 0, 'input', (600, 720), (450, 250), 1, dt = self.dt) 

                #Computing Input layer output spikes and new state
                z, s_input = self.input.forward(x_ts, s_input, batch_size, (self.par.input_kernel, self.par.input_stride, self.par.input_padding))    

                #Buffering output spikes 
                input_buffer, input_buffer_sum = plot.rotate_buffer(z, input_buffer)
                
                #Plotting output spikes of input layer
                plot.plot2D_discrete(input_buffer_sum[:,:,0], self.input.conv_input_dim[0], self.input.conv_input_dim[1], 0, 'downsampled input', (1, 720), (450, 250), 1, dt = self.dt)           

                
                
                #SSConv layer
                z, s_SSConv, self.par.SSConv_v_th, boxes, box_indices, v_decayed, max_indices, stiffness_ave, stiffness_goal = self.SSConv.forward(z, s_SSConv, batch_size, self.SSConv_weights, s_input.X, device, False, time)
                self.SSConv = SSConv(self.par, self.dt)
                
                # #Collect results
                # threshold_buffer.append(self.par.SSConv_v_th[0,0,0,0])
                # stiffness_goal_buffer.append(stiffness_goal)
                # stiffness_ave_buffer.append(stiffness_ave)

                print('Time', time)
                #print("Current v_th value: ", self.par.SSConv_v_th)

                #Buffering output spikes 
                SSConv_buffer, SSConv_buffer_sum = plot.rotate_buffer(z, SSConv_buffer)

                # #Plotting presynaptic traces of SSConv layer
                # plot.plot_presynaptic_traces(s_input.X, self.par.map_colours, 1, 2, boxes, box_indices, False,  0, 'pst_SSConv', (800, 685), (525, 250), 1, device)

                # #Plotting output spikes of SSConv layer in separate windows
                # plot.plot_output_spikes_separate(z, self.par.map_colours, 0,0, 4, 4, 'SSConv_output_spikes_seperate', (527, 685), (525,250), 1, device)

                #Plotting output spikes of SSConv layer in one window
                plot.plot_output_spikes_together(SSConv_buffer_sum, self.par.map_colours, 0, 0, 'SSConv_output_spikes_together', (1000, 685), (525,250), 1, device)

                # #Plot output spikes of SSConv layer in 3D window, very slow only for illustration
                # plot.plot_output_spikes_cube(spike_indices_maps, *self.SSConv.conv_SSConv_dim[:2], self.par.SSConv_out_dim, s, k, self.par.map_colours , 'maps ')
                # plt.show()

                # #Buffering OL_maps
                # OL_dif_windows_buffer, OL_dif_windows_buffer_sum = plot.rotate_buffer(OL_maps_ave, OL_dif_windows_buffer)

                # #plot.plot_voltage_trace(OL_dif_windows_buffer_sum, 0, 0, 'OL per pixel', (500, 300), (500, 300), 1, device)
                # plot.plot_voltage_trace(self.par.SSConv_v_th, 0, 0, 'threshold', (1000, 300), (500, 300), 0, device)
               
               
               
                #Merge layer 
                z, s_merge = self.merge.forward(z, s_merge, batch_size, device)

                #Buffering output spikes 
                merge_buffer, merge_buffer_sum = plot.rotate_buffer(z, merge_buffer)

                #Plotting output spikes of merge layer 
                plot.plot_output_spikes_together(merge_buffer_sum, [(100, 100, 100)],0, 0,  'merge_output_spikes', (0, 0), (525,250), 1, device)
                

                
                #MSConv layer
                z, s_MSConv, self.par.MSConv_v_th, self.par.MSConv_tau_max, boxes, box_indices, v_decayed, max_indices, stiffness_ave, stiffness_goal = self.MSConv.forward(z, s_MSConv, batch_size, s_STDP_exc.weights, s_STDP_inh.weights, s_merge.X, device, training, time, self.STD_goal)
                self.MSConv = MSConv(self.par, self.dt)
                
                #Buffering output spikes 
                MSConv_buffer, MSConv_buffer_sum = plot.rotate_buffer(z, MSConv_buffer)

                # #Before MSConv output can be plotted short run needs to be performed since colord depend on OF magnitudes
                # #Determine current map_colours
                # OF_class = compute_OF(self.par, self.MSConv_weights_exc, self.MSConv_weights_inh, device, -1)
                # OF = OF_class.compute_OF()[0]
                # map_colours = OF_class.assign_MSConv_colour('colour_code.png', OF)
                

                # #Plotting presynaptic traces of MSConv layer
                # plot.plot_presynaptic_traces(s_merge.X, self.par.map_colours, 2, 5, boxes, box_indices, True,  0, 'pst_MSConv', (1, 700), (1000, 300), 1, device)
                
                # #Plotting output spikes of MSConv layer in separate windows
                # plot.plot_output_spikes_separate(z, map_colours, 0, 0, 8, 8, 'MSConv_output_spikes_seperate', (527, 685), (525,250), 1, device)

                # #Plotting output spikes of MSConv layer in one window
                # plot.plot_output_spikes_together(MSConv_buffer_sum, map_colours, 0, 1, 'MSConv_output_spikes_together', (527, 685), (525,250), 0, device)

                #STDP rule
                if training: 
                    s_STDP_exc, X =  self.STDP_exc.forward1(s_merge.X, v_decayed, max_indices, z, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device) , torch.tensor(self.par.MSConv_padding3D).to(device), device, s_STDP_exc)
                    s_STDP_inh, X =  self.STDP_inh.forward1(s_merge.X, v_decayed, max_indices, z, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device) , torch.tensor(self.par.MSConv_padding3D).to(device), device, s_STDP_inh)


                #Plotting MSConv weights
                weights = s_STDP_exc.weights + self.par.MSConv_beta * s_STDP_inh.weights

                # #Plotting all MSConv weights in one window and sort them in increasing order of OF-magnitude
                # #Determine length of optical flow vectors 
                # OF_length = (OF[:, 0]**2 + OF[:, 1]**2)**0.5
                # #Extract sorted indices
                # OF_sorted_idx = np.argsort(OF_length)
                # #Order OF magnitudes
                # OF_length_sorted = np.round(OF_length[OF_sorted_idx], 2)
                # #Plot weights 
                # plot.plot_weights_MSConv_sorted(weights[OF_sorted_idx], 'MSConv_weights_sorted', (1,1),(2000, 1000), True, 0, device)

                # #Plot STD of brightness per delay vs sorted map number
                # STD_sorted_OF = torch.std(self.delay_weights, dim = 1)/torch.mean(self.delay_weights, dim = 1)
                # plt.figure()
                # plt.plot(STD_sorted_OF.cpu())
                # plt.grid()
                # plt.show()

                
                # #Plotting distribution of OF vectors
                # OF_class.plot_OF('colour_code.png', OF)
               
                # #x-tau
                # plot.plot_weights_MSConv(weights[:,:,:,:,0], 8, 8, 'MSConv_weights_x_tau', (1,462),(1050, 190), False, 1, device)
                # #y-tau
                # plot.plot_weights_MSConv(weights[:,:,:,0], 8, 8, 'MSConv_weights_y_tau', (1,462),(1050, 190), False, 1, device)
                #x-y, tau = 0
                plot.plot_weights_MSConv(weights[:,:,0], 8, 8, 'MSConv_weights_x_y_tau0', (1,1),(600, 600), True, 1, device)
                # #x-y, tau = 1
                # plot.plot_weights_MSConv(weights[:,:,1], 8, 8, 'MSConv_weights_x_y_tau1', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 2
                # plot.plot_weights_MSConv(weights[:,:,2], 8, 8, 'MSConv_weights_x_y_tau2', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 3
                # plot.plot_weights_MSConv(weights[:,:,3], 8, 8, 'MSConv_weights_x_y_tau3', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 4
                # plot.plot_weights_MSConv(weights[:,:,4], 8, 8, 'MSConv_weights_x_y_tau4', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 5
                # plot.plot_weights_MSConv(weights[:,:,5], 8, 8, 'MSConv_weights_x_y_tau5', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 6
                # plot.plot_weights_MSConv(weights[:,:,6], 8, 8, 'MSConv_weights_x_y_tau6', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 7
                # plot.plot_weights_MSConv(weights[:,:,7], 8, 8, 'MSConv_weights_x_y_tau7', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 8
                # plot.plot_weights_MSConv(weights[:,:,8], 8, 8, 'MSConv_weights_x_y_tau8', (1,462),(600, 600), True, 1, device)
                # #x-y, tau = 9
                plot.plot_weights_MSConv(weights[:,:,9], 8, 8, 'MSConv_weights_x_y_tau9', (620,1),(600, 600), True, 1, device)

                # #plot exc weight
                # plot.plot_weights_MSConv(s_STDP_exc.weights[:,:,0], 8, 8, 'MSConv_weights_exc_x_y_tau0', (1,1),(600, 600), False, 1, device)
                # plot.plot_weights_MSConv(s_STDP_inh.weights[:,:,0] + 1, 8, 8, 'MSConv_weights_inh_x_y_tau0', (620,1),(600, 600), False, 1, device) 


            print("sequence: ", sequence)
            # print(Li_mean)

            #Save weights
            torch.save(s_STDP_exc.weights, self.weights_name_exc)

            #Save weights
            torch.save(s_STDP_exc.weights, self.weights_name_inh)

    
        return  s_STDP_exc,s_STDP_inh

