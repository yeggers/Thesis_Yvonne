
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
import scipy
import scipy.ndimage as ndi
import tools.plotting as plot
import torch
import torchvision

from functional.STDP_federico2 import STDPFedericoState
from matplotlib.animation import FuncAnimation
from module.lif_mod import LIFModFeedForwardCell, LIFModParameters
from module.noDyn import noDynFeedForwardCell, noDynParameters
from module.STDP_federico import (STDPFedericoFeedForwardCell,
                                  STDPFedericoParameters)
from SNN_param import SNN_param

from models.layer_definitions import Input, SSConv


class TrainSSConv(torch.nn.Module):
    """ This class can be used to train the SSConv layer in Federicos SNN.

    Parameters: 
        device (torch.device): device on which training should be performed (CPU/GPU)
        par (object): SNN parameters
        height (int): number of pixels in input images in vertical direction 
        width (int): number of pixels in input image in horizontal direction 
        method (string): 
        alpha (int): 
        dt(float): time step of the simulation 
    """
    
    def __init__(
        self, device, par, weights_name, method = "super", alpha = 100, dt = 10**(-3), SSConv_weights = "SSConvWeights.pt"
    ):
        super(TrainSSConv, self).__init__()
 
        self.dt = dt
        self.par = par

        #Initializing network layers 
        self.input = Input(self.par, self.dt)
        self.noDyn_input = self.input.noDyn_input

        self.SSConv = SSConv(self.par, self.dt)
        self.lif_SSConv = self.SSConv.lif_SSConv

        #Loading weights 
        self.SSConv_weights = torch.load(SSConv_weights).to(device).to(torch.float32)
        self.weights_name = weights_name

        #Setting up STDP rule 
        self.STDP_params = STDPFedericoParameters(n = self.par.training_n, w_init = self.par.SSConv_w_init, a = self.par.training_a, L = self.par.training_L, ma_len = self.par.training_ma_len)
        self.STDP = STDPFedericoFeedForwardCell(self.par.SSConv_weight_shape, (self.par.SSConv_out_dim, self.SSConv.conv_SSConv_dim[0], self.SSConv.conv_SSConv_dim[1]), p = self.STDP_params, dt = self.dt)


    def forward(self, 
        dir : str,
        batch_size: int, 
        iterations: int,
        training:bool,
        device: torch.device 
    ) -> Tuple[torch.Tensor, STDPFedericoState]:

        """ Compute STDP updates for SSConv layer 

        Parameters: 
            dir (str) : directory containing input sequences
            batch_size (int): number of sequences to be processed at same time
            iterations (int): number of iterations to use during training
            training (bool): specified whether training should be performed
            device (torch.device): device on which computations shall be performed
        """
     
        #Specifying initial state for STDP rule 
        s_STDP = self.STDP.initial_state(batch_size, weights = self.SSConv_weights, device = device, dtype = torch.float32 )

        #Initializing input layer 
        s_input = self.noDyn_input.initial_state(batch_size, device = device, dtype = torch.float32)
        #Initializing SSConv layer 
        s_SSConv = self.lif_SSConv.initial_state_WTA(batch_size, self.par.SSConv_len_stiffness_buffer, device = device, dtype = torch.float32)
         
     
        #Initialize containers to collect results
        stiffness_ave_buffer = []
        threshold_buffer = []
        stiffness_goal_buffer = []

       
        for sequence in range(iterations):
            print("Sequence: ", sequence)
            
            #Randomly selecting sequence from directory 
            random_files = np.random.choice(os.listdir(dir), batch_size)
            
            #Loading first sequence
            #x = torch.load(dir + '/{}'.format(random_files[0]))

            data = bz2.BZ2File(dir + '/{}'.format(random_files[0]), 'rb')
            x = cPickle.load(data)
            
            #Determine sequence length
            seq_length = x.shape[0]
        
            #Adding remaining sequences in batch 
            for sequence in range(1,len(random_files)):
                file_name = dir + '/{}'.format(random_files[sequence])
                load = torch.load(file_name)[:int(seq_length/2)]
                x = torch.cat((x, load),dim = 1)
    
            # #Only taking half of the sequence 
            # x = x[:int(seq_length/2)]
    
            #Randomly decide whether or not to flip image in x and y direction, polarity and time
            flip_w, flip_h, flip_p = tuple(np.random.randint(0,2,3))

            #Apllying flips
            x = (1 - flip_w) * x + flip_w * x[:,:, :,:,  torch.arange(x.shape[4] -1, -1, -1)]
            x = (1 - flip_h) * x + flip_h * x[:, :, :, torch.arange(x.shape[3] -1, -1, -1)]
            x = (1 - flip_p) * x + flip_p * x[:,:, torch.arange(x.shape[2] -1, -1, -1)]
       
            # #Plotting weight distribution 
            # plot.plot_histogram(s_STDP.weights, 500, 0, 1, 3, 'Weights [-]', device)

            #Initialize buffers for plotting 
            input_buffer = torch.zeros(30, batch_size, self.par.input_out_dim, self.par.SSConv_m, *self.input.conv_input_dim[0:2]).to(device)
            SSConv_buffer = torch.zeros(50, batch_size, self.par.SSConv_out_dim, self.par.merge_m, *self.SSConv.conv_SSConv_dim[0:2]).to(device)
            
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
                plot.plot2D_discrete(input_buffer_sum[:, :,0], self.input.conv_input_dim[0], self.input.conv_input_dim[1], 0, 'downsampled input0', (1, 300), (450, 250), 1, dt = self.dt)  
              
                
                
                #SSConv layer
                z, s_SSConv, self.par.SSConv_v_th, boxes, box_indices, v_decayed, max_indices, stiffness_ave, stiffness_goal = self.SSConv.forward(z, s_SSConv, batch_size, s_STDP.weights, s_input.X, device, training, time)
                self.SSConv = SSConv(self.par, self.dt)
               
                # #Collect results, only during short runs since it takes a lot of memory
                # threshold_buffer.append(self.par.SSConv_v_th[0,0,0,0])
                # stiffness_goal_buffer.append(stiffness_goal)
                # stiffness_ave_buffer.append(stiffness_ave[0,0,0,0])
                
                #print('Time', time)
                #print("Current v_th value: ", self.par.SSConv_v_th)
                
                #Buffering output spikes 
                SSConv_buffer, SSConv_buffer_sum = plot.rotate_buffer(z, SSConv_buffer)

                # #Plotting presynaptic traces of SSConv layer
                # image_pst = plot.plot_presynaptic_traces(s_input.X, self.par.map_colours, 1, 2, boxes, box_indices, True,  0, 'pst_test', (0, 400), (960, 500), 1, device)

                # #Plotting output spikes of SSConv layer in separate windows
                # plot.plot_output_spikes_separate(SSConv_buffer_sum, self.par.map_colours, 0, 0, 4, 4, 'SSConv_output_spikes_seperate', (527, 400), (450,250), 1, device)

                #Plotting output spikes of SSConv layer in one window
                image_output = plot.plot_output_spikes_together(SSConv_buffer_sum, self.par.map_colours, 0, 0, 'SSConv_output_spikes_together0', (700, 600), (450,250), 1, device)

                # #Plot output spikes of SSConv layer in 3D window, very slow only for illustration
                # plot.plot_output_spikes_cube(spike_indices_maps, *self.SSConv.conv_SSConv_dim[:2], self.par.SSConv_out_dim, s, k, self.par.map_colours , 'maps ')
                # plt.show()

            
                #Training
                if training:
                    #Performing STDP rule
                    s_STDP, X =  self.STDP.forward1(s_input.X, v_decayed, max_indices, z, torch.tensor(self.par.SSConv_kernel3D).to(device), torch.tensor(self.par.SSConv_stride3D).to(device) , torch.tensor(self.par.SSConv_padding3D).to(device), device, s_STDP)

                #Plotting SSConv weights
                image_weights = plot.plot_weights_SSConv(s_STDP.weights, 'SSConv_weights_after', (1,1),(1050, 190), True, 1, device )

                # #Plotting colour legend for SSConv layer 
                # plot.plot_SSConv_colour_legend(s_STDP.weights, 'SSConv_colour_legend', (700,400),(1050, 190), True, 0, device, self.par.map_colours)

                
                # #Save weights as PNG
                # if time in [10, 500, 1000, 2000, 5000, 10000, 30000, 60000, 80000, 100000, 120000, 150000, 180000, 200000, 240000]:
                #     image_weights = cv2.resize(image_weights, (0,0), fx = 15, fy = 15, interpolation = cv2.INTER_NEAREST )
                #     # image_output = cv2.resize(image_output, (0,0), fx = 15, fy = 15, interpolation = cv2.INTER_NEAREST )
                #     # image_pst = cv2.resize(image_pst, (0,0), fx = 15, fy = 15, interpolation = cv2.INTER_NEAREST )

                #     cv2.imwrite('images/{par_name}/vth/vth_{threshold}/weights_alpha_{alpha}_vth_{threshold}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_t_{time}.png'.format(par_name = self.par.par_name, threshold = self.par.SSConv_v_th, alpha = self.par.SSConv_alpha, lambda_X = self.par.SSConv_lambda_X, lambda_v = self.par.SSConv_lambda_v, time = time), image_weights)
                   
        
            #Save weights
            torch.save(s_STDP.weights, self.weights_name)

            # with bz2.BZ2File("pickles/{par_name}/stiffness/vth/stiffness_alpha_{alpha}_vth_{v_th}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_iterations_{iterations}.pbz2".format(par_name = self.par.par_name, iterations = iterations, alpha = self.par.SSConv_alpha, v_th = self.par.SSConv_v_th, lambda_X = self.par.SSConv_lambda_X, lambda_v = self.par.SSConv_lambda_v), 'w') as f: 
            #     cPickle.dump(stiffness_ave, f)
            
            #Print convergence parameter
            #print('Li_cur_buffer: ', s_STDP.Li_buffer)
            print("Li_cur: ", torch.mean(s_STDP.Li_buffer, dim = 0))



        # #Plot stiffness history
        # plt.figure()
        # plt.plot(stiffness_ave_buffer, label = 'stiffness moving average')
        # plt.plot(stiffness_goal_buffer, label = 'target')
        # plt.title('Stiffness history')
        # plt.xlabel("Time step [-]")
        # plt.ylabel("$\mathrm{OL}_{\mathrm{WTA}}$ [-]")
        # plt.grid()
        # plt.legend()
        # #plt.savefig('images/{par_name}/vth/vth_{threshold}/stiffness_history_alpha_{alpha}_vth_{threshold}_lambda_X_{lambda_X}_lambda_v_{lambda_v}.pdf'.format(par_name = self.par.par_name, threshold = self.par.SSConv_v_th, alpha = self.par.SSConv_alpha, lambda_X = self.par.SSConv_lambda_X, lambda_v = self.par.SSConv_lambda_v))

        # #Plot voltage threshold history
        # plt.figure()
        # plt.plot(threshold_buffer, label  = 'thresholds')
        # plt.title("History of $v_{th}$")
        # plt.xlabel("Time step [-]")
        # plt.ylabel('$v_{th}$ [ms]')
        # plt.grid()
        # plt.legend()
        # #plt.savefig("images/thresholds_alpha_{alpha}_v_th_adaptive_lambda_{lambda_X}.pdf".format(alpha = self.par.SSConv_alpha, v_th = self.par.SSConv_v_th, lambda_X = self.par.SSConv_lambda_X))

    
        # plt.show()
               
        return s_STDP



 

    


 