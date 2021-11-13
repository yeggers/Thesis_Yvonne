
import os
import random
import time
from typing import NamedTuple, Tuple

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
from functional.lif_mod2 import get_max_pst
from module.noDyn import noDynFeedForwardCell, noDynParameters
from module.STDP_federico import (STDPFedericoFeedForwardCell,
                                    STDPFedericoParameters)
from functional.STDP_federico2 import STDPFedericoState

from models.layer_definitions_noSSConv import Input, Merge, MSConv


class TrainMSConv_noSSConv(torch.nn.Module):
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
        self, device, par, height = 180, width = 240, dt = 10**(-3), MSConvWeights_exc = "MSConvWeights_exc.pt", MSConvWeights_inh = "MSConvWeights_inh.pt"
    ):
        super(TrainMSConv_noSSConv, self).__init__()

   
        self.dt = dt
        self.par = par

        #Initializing network layers 
        self.input = Input(self.par, self.dt)
        self.noDyn_input = self.input.noDyn_input

        self.merge = Merge(self.par, self.dt)
        self.lif_merge = self.merge.lif_merge

        self.MSConv = MSConv(self.par, self.dt)
        self.lif_MSConv = self.MSConv.lif_MSConv

        #Loading weights 
        self.MSConv_weights_exc = torch.load(MSConvWeights_exc).to(device)
        self.MSConv_weights_inh = torch.load(MSConvWeights_inh).to(device)

        #Setting up STDP rule 
        self.STDP_params_exc = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.MSConv_w_init_ex, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP_exc = STDPFedericoFeedForwardCell(self.par.MSConv_weight_shape, (self.par.MSConv_out_dim, self.MSConv.conv_MSConv_dim[0], self.MSConv.conv_MSConv_dim[1]), p = self.STDP_params_exc, dt = self.dt)

        self.STDP_params_inh = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.MSConv_w_init_inh, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP_inh = STDPFedericoFeedForwardCell(self.par.MSConv_weight_shape, (self.par.MSConv_out_dim, self.MSConv.conv_MSConv_dim[0], self.MSConv.conv_MSConv_dim[1]), p = self.STDP_params_inh, dt = self.dt)


    def forward(self,  
        dir : str,
        batch_size:int,
        iterations: int,
        device: torch.device 
    ) -> Tuple[torch.Tensor, STDPFedericoState]:

        
        #"""Compute STDP updates for SSConv layer 

        # Parameters: 
        #     dir (str) : directory containing input sequences if random_file = True
        #     batch_size (int): number of sequences to be processed at same time
        #     iterations (int): number of iterations to use during training
        #     device (torch.device): device on which computations shall be performed
        # """
      
        
        #Specifying initial state for STDP rule 
        s_STDP_exc = self.STDP_exc.initial_state(batch_size, self.MSConv_weights_exc, device = device, dtype = torch.float32 )
        s_STDP_inh = self.STDP_inh.initial_state(batch_size, self.MSConv_weights_inh, device = device, dtype = torch.float32 )
         
        #Initializing epoch counter 
        epoch = 0

        #Initializing while loop boolean 
        enter_loop = True 
    
        #while enter_loop:
        for epoch in range(iterations):
            
            #Incrementing epoch counter
            epoch += 1
           
            #Initializing input layer 
            s_input = self.noDyn_input.initial_state(batch_size, device = device, dtype = torch.float32)
            #Initializing Merge layer 
            s_merge = self.lif_merge.initial_state_NT(batch_size, device = device, dtype = torch.float32)
            #Initializing MSConv layer 
            s_MSConv = self.lif_MSConv.initial_state_WTA(batch_size, device = device, dtype = torch.float32)

        
            #Randomly selecting sequence from directory 
            random_files = random.sample(os.listdir(dir), batch_size)
            
            #Loading first sequence
            x = torch.load(dir + '/{}'.format(random_files[0]))

            #Adding remaining sequences in batch 
            for sequence in range(1,len(random_files)):
                file_name = dir + '/{}'.format(random_files[sequence])
                load = torch.load(file_name)
                x = torch.cat((x, load),dim = 1)
    
            #Determine sequence length
            seq_length = x.shape[0]
      
            # #Randomly decide whether or not to flip image in x and y direction and polarity
            # flip_w, flip_h, flip_p = tuple(np.random.randint(0,2,3))
       

            # #Apllying flips
            # x = (1 - flip_w) * x + flip_w * torch.flip(x, [4])
            # x = (1 - flip_h) * x + flip_h * torch.flip(x, [3])
            # x = (1 - flip_p) * x + flip_p * torch.flip(x, [2])

            # #Plotting weight distribution 
            # plot.plot_histogram(s_STDP_exc.weights + self.par.MSConv_beta*s_STDP_inh.weights, 500, -1, 1, 3, 'Weights_exc [-]', device)
           
            
            #Initializing buffers for plotting 
            input_buffer = torch.zeros(5, batch_size, self.par.input_out_dim, self.par.merge_m, *self.input.conv_input_dim[0:2]).to(device)
            merge_buffer = torch.zeros(50, batch_size, self.par.merge_out_dim, self.par.MSConv_m, *self.merge.conv_merge_dim[0:2]).to(device)
            MSConv_buffer = torch.zeros(50, batch_size, self.par.MSConv_out_dim, self.par.pooling_m, *self.MSConv.conv_MSConv_dim[0:2]).to(device)
            #i_buffer = torch.zeros(seq_length, batch_size, self.par.MSConv_out_dim, self.par.pooling_m, *self.MSConv.conv_MSConv_dim[0:2]).to(device)
            num_spike_layer_buffer = torch.zeros(seq_length, self.par.MSConv_out_dim)
            pst_spikes_per_delay_sum = torch.zeros(10).to(device)
            delays = []
            STD_spike_weights = []
            STD_weights = []
            variances_PST = []

            
            
            for ts in range(seq_length):
                #Only putting input at current time step on GPU
                x_ts = x[ts, :].to(device)  
                
                # #Plotting output spikes of input layer
                # plot.plot2D_discrete(x_ts, self.par.height, self.par.width, 0, 'input', (500, 720), (450, 250), 1, dt = self.dt)

                #Computing Input layer output spikes and new state
                z, s_input = self.input.forward(x_ts, s_input, batch_size, (self.par.input_kernel, self.par.input_stride, self.par.input_padding))    

                # #Buffering output spikes 
                # input_buffer, input_buffer_sum = plot.rotate_buffer(z, input_buffer)
                
                #Plotting output spikes of input layer
                plot.plot2D_discrete(z[:,:,0], self.input.conv_input_dim[0], self.input.conv_input_dim[1], 0, 'downsampled input', (1, 720), (450, 250), 1, dt = self.dt)           

                #Merge layer 
                z, s_merge = self.merge.forward(z, s_merge, batch_size, device)

                # #Buffering output spikes 
                # merge_buffer, merge_buffer_sum = plot.rotate_buffer(z, merge_buffer)

                # #Plotting output spikes of merge layer 
                # plot.plot_output_spikes_together(z, [(100, 100, 100)],0, 0,  'merge_output_spikes', (0, 0), (525,250), 1, device)
                

                
                #MSConv layer
                z, s_MSConv, boxes, box_indices, i = self.MSConv.forward(z, s_MSConv, batch_size, s_STDP_exc.weights, s_STDP_inh.weights, s_merge.X, device, True)
               # i_buffer[ts] = i
                
                map_indices = torch.nonzero(z)[:,1]

                
                # # #Buffering output spikes 
                # MSConv_buffer, MSConv_buffer_sum = plot.rotate_buffer(z, MSConv_buffer)

                #Determine current map_colours
                OF_class = compute_OF(self.par, self.MSConv_weights_exc, self.MSConv_weights_inh, device, 0.6)
                OF = OF_class.compute_OF()[0]
                map_colours = OF_class.assign_MSConv_colour('colour_code.png', OF)
                #OF_class.plot_OF('colour_code.png', OF)

                # #Plotting presynaptic traces of MSConv layer
                # plot.plot_presynaptic_traces(s_merge.X, self.par.map_colours, 2, 5, boxes, box_indices, True,  0, 'pst_MSConv', (1, 300), (1000, 300), 1, device)
                
                # #Plotting output spikes of MSConv layer in separate windows
                # plot.plot_output_spikes_separate(z, map_colours, 0, 0, 8, 8, 'MSConv_output_spikes_seperate', (527, 685), (525,250), 1, device)

                #Plotting output spikes of MSConv layer in one window
                #plot.plot_output_spikes_together(MSConv_buffer_sum, map_colours, 0, 0, 'MSConv_output_spikes_together', (452, 720), (525,250), 1, device)

                #Determine length of optical flow vectors 
                OF_length = (OF[:, 0]**2 + OF[:, 1]**2)**0.5
                #Determine order of increasing OF magnitude 

                OF_sorted_idx = np.argsort(OF_length)
                # OF_length_sorted = np.round(OF_length[OF_sorted_idx], 2)
                # plt.figure()
                # plt.plot(OF_length_sorted)
                # plt.title("Length of OF vector per map")
                # plt.xlabel("Map in ascending order of OF magnitude")
                # plt.ylabel("Magnitude of OF vector")
                # plt.show()
                # #Sorting output spikes inincreasing order of OF magnitude 
                # z_sum_before_sorting =  torch.sum(z, dim = (0,2,3,4))
                # z_sum_after_sorting = torch.sum(z, dim = (0,2,3,4))[OF_sorted_idx]
                #Count number of spikes per OF map 
                #num_spike_layer_buffer[ts] = torch.sum(z, dim = (0,2,3,4))[OF_sorted_idx]

                # #STDP rule 
                #TODO: do not have to extract presyaptic trace twice for inh and exc
                s_STDP_exc, X =  self.STDP_exc.forward1(s_merge.X, z, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device) , torch.tensor(self.par.MSConv_padding3D).to(device),device, s_STDP_exc)
                s_STDP_inh, X =  self.STDP_inh.forward1(s_merge.X, z, torch.tensor(self.par.MSConv_kernel3D).to(device), torch.tensor(self.par.MSConv_stride3D).to(device) , torch.tensor(self.par.MSConv_padding3D).to(device),device, s_STDP_inh)

                # Li_mean = torch.mean(s_STDP_exc.Li_buffer, dim = 0)[torch.where(s_STDP_exc.nu_spikes > 0)]

                #Plotting MSConv weights
                weights = s_STDP_exc.weights + self.par.MSConv_beta * s_STDP_inh.weights

                #Compute STD over delays for spiking maps
                delay_weights_spikes = torch.sum(weights[map_indices] , dim = (1,3,4))
                delay_weights_spikes_STD = torch.std(delay_weights_spikes, dim = 1)/torch.mean(delay_weights_spikes, dim = 1)
                STD_spike_weights.append(torch.mean(delay_weights_spikes_STD))

                #Compute STD over delays for all maps 
                delay_weights = torch.sum(weights , dim = (1,3,4))
                delay_weights_STD = torch.std(delay_weights, dim = 1)/torch.mean(delay_weights, dim = 1)
                STD_weights.append(torch.mean(delay_weights_STD)) 
                
                # plt.figure()
                # plt.plot(np.array(delay_weights_STD[OF_sorted_idx].cpu()))
                # plt.grid()
                # plt.xlabel('Map number [-]')
                # plt.ylabel('$\mathrm{STD}_\mathrm{d}$ [-]')
                # plt.show()

                # #Compute STD of kernel histograms 
                # lambda_weights_x = torch.sum(weights[:, :, 0], dim = (2))
                # lambda_weights_x_STD = torch.std(lambda_weights_x, dim = (2))/ torch.mean(lambda_weights_x, dim = (2))
                # lambda_weights_y = torch.sum(weights[:, :, 0], dim =(3))
                # lambda_weights_y_STD = torch.std(lambda_weights_y, dim = (2))/ torch.mean(lambda_weights_y, dim = (2))

                # #Compute sum of spike distribution per delay
                # pst_spikes_per_delay = torch.mean(X, dim = (0,1,3,4))
                # pst_spikes_STD = torch.std(pst_spikes_per_delay)/torch.mean(pst_spikes_per_delay)
                # variances_PST.append(pst_spikes_STD)

                # #Compute STD of PST histograms
                # pst_spikes_lambda_x = torch.sum(X[:,:,0], dim = (2))
                # pst_spikes_x_STD = torch.std(pst_spikes_lambda_x, dim = (2))/torch.mean(pst_spikes_lambda_x, dim = (2))
                # pst_spikes_lambda_y = torch.sum(X[:,:,0], dim = (3))
                # pst_spikes_y_STD = torch.std(pst_spikes_lambda_y, dim = (2))/torch.mean(pst_spikes_lambda_x, dim = (2))
               
                # plot.plot_weights_MSConv_sorted(weights[OF_sorted_idx], 'MSConv_weights_sorted', (1,1),(2000, 1000), True, 0, device)
                
                
                # plt.figure()
                # plt.plot(self.par.MSConv_delay, np.array(torch.sum(weights[OF_sorted_idx][-1].cpu(), dim = (0,2,3)).cpu()))
                # plt.xlabel("Delay [ms]")
                # plt.ylabel("Sum of weights [-]")
                # plt.ylim((4, 7.5))
                # plt.grid()
                # plt.show()

               

      
                
                
                if ts > self.par.MsConv_tau_max:
                   
                    #Compute delay update 
                    d_tau_max = (torch.mean(delay_weights_STD) - torch.mean(delay_weights_spikes_STD)) *self.par.MsConv_tau_max *1.5
                    
                    # # #Increase buffer length if maximum delays increases
                    # # if d_tau_max > 0:
                    # #     s_merge.buffer = torch.nn.functional.pad(s_merge.buffer, (0,int(d_tau_max*10**(-3)/self.dt) + 1))
                    
                    #Upadte merge parameters with new delays

                    self.par.MsConv_tau_max = self.par.MsConv_tau_max + d_tau_max
                    self.merge = Merge(self.par, self.dt)
                    # print("New tau: ", self.par.MsConv_tau_max)

                    
                    
                    delays.append(self.par.MsConv_tau_max)
                   


                    if len(map_indices)>0:
                        test = 2
                        
                        # plot.plot_weights_MSConv_sorted(weights[map_indices], 'MSConv_weights_sorted', (1,1),(900, 1000), True, 0, device)

                        # plt.figure()
                        
                        # plt.plot(np.array((pst_spikes_per_delay).cpu()))
                        # plt.title("Sum of PST per delays")
                        # plt.xlabel("Delay number [-] ")
                        # plt.ylabel("Sum of PST")
                        # plt.grid()
                        # #plt.pause(1)
                    

                        # plt.figure()
                        # plt.plot(np.array(delay_weights.cpu()))
                        # plt.title("Sum of weights per delay")
                        # plt.xlabel("Delay number [-]]")
                        # plt.ylabel("Sum of weights")
                        # plt.grid()
                        # plt.show()

                

                
                # pst_spikes_per_delay_sum += pst_spikes_per_delay
                #pst_spikes_STD_sum = torch.std(pst_spikes_per_delay_sum)/torch.mean(pst_spikes_per_delay_sum)

                #Sort weights by magnitude of corresponding OF vectors 
                #weights_sorted = weights[OF_sorted_idx]
               
                #plt.figure()
                # for map in range(weights_sorted.shape[0]):
                #     plt.subplot(8,8,map + 1)
                #     plt.plot(np.array(torch.sum(weights_sorted[map].cpu(), dim = (0,2,3)).cpu()))
                #     plt.title("Map {map}".format(map = map))
                #     plt.xlabel("Delay number [-]")
                #     plt.ylabel("weight sum [-]")
                #     plt.ylim((4, 7.5))
                #     plt.grid()
                # plt.show()

                

                # #Compute variance of weight distribution along delays per map 
                # test = torch.sum(weights_sorted, dim = (1,3,4))
                # OF_var = torch.std(torch.sum(weights_sorted, dim = (1,3,4)), dim = 1)/torch.mean(torch.sum(weights_sorted, dim = (1,3,4)), dim = 1)
                # plt.plot(np.array(OF_var.cpu()))
                # #plt.title("Variance of sum of weights per delay for different MSConv maps")
                # plt.xlabel("MSConv map number [-]")
                # plt.ylabel("Normalized STD [-]")
                # plt.grid()
                # plt.show()

              
                # #x-tau
                # plot.plot_weights_MSConv(weights[:,:,:,:,0], 4, 4, 'MSConv_weights_x_tau', (1,462),(1050, 190), True, 1, device)
                # #y-tau
                # plot.plot_weights_MSConv(weights[:,:,:,0], 4, 4, 'MSConv_weights_y_tau', (1,462),(1050, 190), True, 1, device)
                #x-y, tau = 0
                #plot.plot_weights_MSConv(weights[:,:,0], 8, 8, 'MSConv_weights_x_y_tau0', (1,1),(500, 500), True, 1, device)
                # #x-y, tau = 1
                # plot.plot_weights_MSConv(weights[:,:,1], 4, 4, 'MSConv_weights_x_y_tau1', (300,1),(300, 300), True, 1, device)
                # #x-y, tau = 2
                # plot.plot_weights_MSConv(weights[:,:,2], 4, 4, 'MSConv_weights_x_y_tau2', (600,1),(300, 300), True, 1, device)
                # #x-y, tau = 3
                # plot.plot_weights_MSConv(weights[:,:,3], 4, 4, 'MSConv_weights_x_y_tau3', (900,1),(300, 300), True, 1, device)
                # #x-y, tau = 4
                # plot.plot_weights_MSConv(weights[:,:,4], 4, 4, 'MSConv_weights_x_y_tau4', (1200,1),(300, 300), True, 1, device)
                # #x-y, tau = 5
                # plot.plot_weights_MSConv(weights[:,:,5], 4, 4, 'MSConv_weights_x_y_tau5', (1, 300),(300, 300), True, 1, device)
                # #x-y, tau = 6
                # plot.plot_weights_MSConv(weights[:,:,6], 4, 4, 'MSConv_weights_x_y_tau6', (300,300),(300, 300), True, 1, device)
                # #x-y, tau = 7
                # plot.plot_weights_MSConv(weights[:,:,7], 4, 4, 'MSConv_weights_x_y_tau7', (600,300),(300, 300), True, 1, device)
                # #x-y, tau = 8
                # plot.plot_weights_MSConv(weights[:,:,8], 4, 4, 'MSConv_weights_x_y_tau8', (900, 300),(300, 300), True, 1, device)
                #x-y, tau = 9
                #plot.plot_weights_MSConv(weights[:,:,9], 8, 8, 'MSConv_weights_x_y_tau9', (1200, 300),(500, 500), True, 1, device)

                # #plot exc weight
                # plot.plot_weights_MSConv(s_STDP_exc.weights[:,:,0], 8, 8, 'MSConv_weights_exc_x_y_tau0', (1,1),(600, 600), False, 1, device)
                # plot.plot_weights_MSConv(s_STDP_inh.weights[:,:,0] + 1, 8, 8, 'MSConv_weights_inh_x_y_tau0', (620,1),(600, 600), False, 1, device) 


            

              
                
               
                #Computing stopping criterion 
                #Computing the moving average of neurons that have spiked before 
                # Li_mean = torch.mean(torch.mean(s_STDP_exc.Li_buffer, dim = 0)[torch.where(s_STDP_exc.nu_spikes > 0)])
                #Checking if stopping criterion is fulfilled 
                #enter_loop = ~torch.all(torch.ge(self.par.training_L.to(device), Li_mean))
                 
            print("Epoch: ", epoch)
            # print(Li_mean)

            #Plot histogram of input currents 
            #plot.plot_histogram(i_buffer, 100, torch.min(i_buffer), torch.max(i_buffer), 0, 'input_current', device)
            
            # print("variance: ", torch.std(pst_spikes_per_delay_sum)/torch.mean(pst_spikes_per_delay_sum))
            
            # plt.plot(torch.sum(num_spike_layer_buffer, dim = 0))
            # plt.title("Number of spikes per map tau_max = 75 ms")
            # plt.xlabel("Map number")
            # plt.ylabel("Number of spikes")
            # plt.grid()
            # #plt.pause(1)
            

            # plt.figure()
            # plt.plot(np.array((pst_spikes_per_delay_sum).cpu()))
            # plt.title("Distribution of spikes over delays tau_max = 75 ms")
            # plt.xlabel("Delay number")
            # plt.ylabel("Number of spikes")
            # plt.grid()
            # #plt.pause(1)
            

            plt.figure()
            plt.plot(delays)
            #plt.title("History of tau_max")
            plt.xlabel("Time step [-]")
            plt.ylabel(r'$\tau_{max}$ [ms]')
            plt.grid()


            # plt.figure()
            # plt.plot(variances_weights, 'r')
            # plt.plot(variances_PST, 'g')
            # plt.title("History of STD")
            # plt.xlabel("Time step [-]")
            # plt.ylabel("STD [-]")
            # plt.grid()
            # plt.legend("STD weights", "STD PST")

            
            # plt.figure()
            # plt.plot(np.array((pst_spikes_per_delay).cpu()))
            # plt.title("Distribution of spikes over delays")
            # plt.xlabel("Delay number")
            # plt.ylabel("Number of spikes")
            # plt.grid()

            plt.figure()
            plt.plot(STD_spike_weights)
            plt.plot(STD_weights)
            #plt.title("Mean of STDs")
            plt.xlabel("Time step[-]]")
            plt.ylabel("$\overline{\mathrm{STD}}_{d}^{N}$")
            plt.grid()
           
            
            plt.show()
        

            
        #   if len(Li_mean)>0:
        #         plot.plot_histogram(Li_mean, 500, 0, 1, 0, 'Li_mean [-]', device)
        
        return s_STDP_exc,s_STDP_inh

