

import torch
import cv2
import time 
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt

from typing import NamedTuple, Tuple

import tools.plotting as plot


class STDPFedericoParameters(NamedTuple):
    '''Parameters of Federicos STDP learning rule.

    Parameters:
        n (torch.Tensor): Learning rate 
        w_init (torch.Tensor): Value of inital  weights
        a (torch.Tensor): Parameter determning spread of weights 
        L (torch.Tensor): Convergence threshold
        ma_len(torch.Tensor): Length of moving average window for stopping criterion
    '''

    n: torch.Tensor = torch.as_tensor(10**(-4))
    w_init: torch.Tensor = torch.as_tensor(0.5)
    a: torch.Tensor = torch.as_tensor(0)
    L: torch.Tensor = torch.as_tensor(5*10**(-2))
    ma_len: torch.Tensor = torch.as_tensor(10)


class STDPFedericoState(NamedTuple):
    """State Federicos STDP learning rule.

    Parameters:
        weights (torch.Tensor): Weighs of the neural layers
        Li_buffer (torch.Tensor): Buffer to compute moving average of stopping criterion
        nu_spikes (torch.Tensor): Counter to determine how often a neuron has spiked
    """

    weights: torch.Tensor
    Li_buffer: torch.Tensor
    nu_spikes: torch.Tensor 


def stdp_federico_step(
    x: torch.Tensor,
    z: torch.Tensor,
    s:int,
    device, 
    state: STDPFedericoState,
    shape: tuple,
    p: STDPFedericoParameters = STDPFedericoParameters(),
    dt: float = 10**(-3),
) -> Tuple[torch.Tensor, STDPFedericoState]:
    """Federicos STDP learning rule.

    Parameters:
        x(torch.Tensor): presynaptic spike traces
        z(torch.Tensor): postsynaptic spikes
        s(int): Stride used in layer to be trained 
        state (STDPFedericoState): state of the STDP 
        shape (tuple): shape of the weights of the current layer
        p (STDPFedericoParameters): STDP parameters
        dt (float): integration time step
    """
    
    #Extacting presynaptic traces from spike history of neurons in previous layer 
    #Creating array in which presynaptic traces of all neurons are collected. 
    #The dimensions of the array are [image height, image width, number of in channels, number of synapses between two neurons, kernel size, kernel size]
    #TODO: Make it work for batch size larger than one
    #TODO: Account for padding 
    
    #
    start_time = time.time()
    spike_indices = torch.nonzero(z)
    
    # #Finding maximum synapse trace: for each neuron 
    # for h in range(z.shape[-2]): 
    #     for w in range(z.shape[-1]):
    #         #Collecting presynaptic traces for each neuron from spiking activity of neurons in the previous layer
    #         test = x[:, :, :, s*h: s*h + shape[-1], s*w:s*w+ shape[-1]]
    #         X[:,h, w] = x[:, :, :, s*h: s*h + shape[-1], s*w:s*w+ shape[-1]]
    #         X_max = torch.amax(X[:,h,w], dim = (1,2,3,4)) 
    #         X_min = torch.amin(X[:,h,w], dim = (1,2,3,4))
            
    #         if X_max != 0:
    #             X[:, h, w] = (X[:, h, w] - X_min)/X_max
    #         #X[:, h, w] = torch.tensor([(X[i, h, w] - X_min[i])/X_max[i] for i in range(len(X_max)) if X_max[i] != 0])
    
    
    # #Expanding per neuron preysnaptic traces to include number of channels in dimensions
    # X = X.unsqueeze(3).expand( -1, -1,-1, shape[0], -1, -1, -1, -1)

    # #Expanding weights to be defined for each neuron (rather than each layer) to allow for multiplication of weights and presynaptic traces
    # W = state.weights.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(z.shape[0],z.shape[-2], z.shape[-1], -1, -1, -1, -1, -1)

    k = shape[-1]   #Kernel size 
    X = torch.zeros(len(spike_indices), *shape[1:], device = device, dtype = x.dtype)
    dW = torch.zeros(len(spike_indices), *shape, device = device, dtype = x.dtype)
    Li = torch.ones(2,2).to(device)
    W_new = state.weights


    
    
    #Maybe use unfold for this
    
    if len(spike_indices)> 0:
        for i in range(len(spike_indices)):
        
            #Getting presynaptic spike trace of current spiking neuron from spike history in previous layer
            X[i] = x[spike_indices[i,0], :,:, s*spike_indices[i,3]: s*spike_indices[i,3] + k , s*spike_indices[i,4]: s*spike_indices[i,4]+k]
            
            h = np.arange(x.shape[-2])
            w = np.arange(x.shape[-1])
            plt.clf()
            for out_dim in range(x.shape[1]):
                plt.subplot(1, x.shape[1], out_dim + 1)
                plt.title("presynaptic traces")
                plt.pcolormesh(w, h, x[0,out_dim,0].cpu(), cmap=plt.cm.get_cmap('Blues'))
                plt.colorbar()
                plt.ylim([x.shape[-2], 0])
                plt.title("Spike trains of neurons in the input layer")
            
            
            
            
            plt.figure()
            h = np.arange(X.shape[-2])
            w = np.arange(X.shape[-1])
            for in_dim in range(X.shape[1]):
                for m in range(X.shape[2]):
                   pst = X[i, in_dim, m].cpu()
                   plt.subplot(X.shape[1], X.shape[2], in_dim*X.shape[2] +m +1)
                   plt.title("Presynaptic spike traces in channel "+ str(in_dim) + " and synapse " + str(m) )
                   plt.pcolormesh(w, h, pst, cmap=plt.cm.get_cmap('Blues'))
                   plt.ylim([X.shape[-2]-1,0])
                   plt.colorbar()
            plt.show()
            
            
            #Normalizing spike trace 
            X[i] = (X[i] - torch.min(X[i]))/ (torch.max(X[i]) - torch.min(X[i]))

            #Weights for map of current spike 
            W = state.weights[spike_indices[i,1]]
        
            #Computing weight update 
            LTP_X = torch.exp(X[i]) - p.a
            LTP_W = torch.exp(-(W - p.w_init))
            LTP = LTP_W*LTP_X

            LTD_X = torch.exp(1 - X[i]) - p.a
            LTD_W = -torch.exp(W - p.w_init)
            LTD = LTD_W*LTD_X

            dW[i, spike_indices[i,1]] = p.n*(LTP + LTD)

          
           
            
            
        #Taking average of weights from all neurons
        dW = torch.mean(dW, dim = (0))
        # plt.figure()
        # for out_dim in range(shape[0]):
        #     for in_dim in range(shape[1]):
        #         pst = dW[out_dim, in_dim, 0].cpu()
        #         plt.subplot(shape[0], shape[1], out_dim*shape[1] +in_dim +1)
        #         plt.title("dW in out-channel" + str(out_dim)+ "in-channel "+ str(in_dim) )
        #         plt.pcolormesh(h, w, pst, cmap=plt.cm.get_cmap('Blues'))
        #         plt.colorbar()
        # plt.show()

       
       
        #Computing updates
        W_new = state.weights + dW
        #Plotting new weights 
        
        #ani = FuncAnimation(plt.gcf(), animateW(shape, W_new, h, w), 1000)
        #plt.tight_layout()
        #plt.show()
       
        

        #Computing stopping criterion
        W_mean = (W_new - torch.min(W_new))/(torch.max(W_new) - torch.min(W_new))
        Li = 1/(shape[1]*shape[2]*shape[3]**2)*torch.sum((X-W_mean[spike_indices[:,1]])**2, dim = (1, 2,3,4))
       
   

    

    #print ("Learning took ", time.time() - start_time, "to run")

    return Li, (STDPFedericoState(W_new))


def stdp_federico_step2(
    x: torch.Tensor,
    v_decayed,
    max_indices: torch.Tensor,
    z: torch.Tensor,
    kernel: torch.Tensor,
    stride: torch.Tensor,
    pad: torch.Tensor, 
    device: torch.device, 
    state: STDPFedericoState,
    shape: tuple,
    p: STDPFedericoParameters = STDPFedericoParameters(),
    dt: float = 10**(-3),
    
) -> Tuple[torch.Tensor, STDPFedericoState]:
    """Federicos STDP learning rule.

    Parameters:
        x(torch.Tensor): presynaptic spike traces
        max_indices (torch.Tensor): indices of maximum prsynaptic trace per neural neighbourhood
        z(torch.Tensor): postsynaptic spikes
        s(int): Stride used in layer to be trained 
        state (STDPFedericoState): state of the STDP 
        shape (tuple): shape of the weights of the current layer
        p (STDPFedericoParameters): STDP parameters
        dt (float): integration time step
    """

    ##Determine start time to check how long computations take
    #start_time = time.time()
    
    #Initializing states 
    W_new = state.weights
    W_mean = state.weights
    Li_cur = torch.tensor([0.])
    Li_buffer_new = state.Li_buffer
    nu_spikes_new = state.nu_spikes
    X = torch.zeros(1, x.shape[1], x.shape[2], kernel[1], kernel[2]).to(device)
    x_unfolded_max = x

    #Determine indices of spikes 
    spike_indices = torch.where(z == 1)
   

    if len(spike_indices[0])>0:
        #Apply padding to x 
        x = torch.nn.functional.pad(x, (pad[2], pad[2], pad[1], pad[1], pad[0], pad[0]))

        #Unfold tensor containing spike history of previous layer to obtain presynaptic spike train of all neurons
        #Result is of dimension [shape of MSConv output [N, channels (1 since all channels have the same pst), number of delays (1 since they disappear in the convolution), h, w], dimensions of PST ([ch, m, k, k])]
        x_unfolded = x.unfold(1, x.shape[1], 1).unfold(2, kernel[0], stride[0]).unfold(3, kernel[1], stride[1]).unfold(4, kernel[2], stride[2])

        #Expanding unfolded x-tensor to number of output channels in current layer (presynaptic traces are the same for all channels)
        x_unfolded = x_unfolded.expand(-1, shape[0], -1, -1, -1, -1, -1, -1, -1)

        #Extracting presynaptic spike traces of all spiking neurons 
        X = x_unfolded[spike_indices]

        #Normalizing pressyaptic spike train
        Xmax = torch.amax(X, dim = (1,2,3,4), keepdim=True)
        Xmin = torch.amin(X, dim = (1,2,3,4), keepdim=True)
        X = (X - Xmin)/(Xmax - Xmin)

        # #Plotting presynaptic trace windows
        # plot.plot_pst_windwos(X, spike_indices, 'pst_windows', (1300, 1),(900, 1000), 1, device)

        #Extracting current weights of all spiking neurons
        W = state.weights[spike_indices[1]]

        #Computing weight update
        LTP_X = torch.exp(X) - p.a
        LTP_W = torch.exp(-(W - p.w_init))
        LTP = LTP_W*LTP_X

        LTD_X = torch.exp(1 - X) - p.a
        LTD_W = -torch.exp(W - p.w_init)
        LTD = LTD_W*LTD_X

        #Distributing weights over the corresponding sequences and channels 
        dW = torch.zeros(len(spike_indices[0]), z.shape[0],*state.weights.shape).to(device)
        dW[tuple(torch.arange(dW.shape[0])), spike_indices[0], spike_indices[1]] = p.n*(LTP + LTD) 

        #Taking channel-wise average of weights 
        dW = torch.mean(dW, dim = 0)

        #Summing up contributions of different sequences 
        dW = torch.sum(dW, dim = 0)

        # #Plotting dW 
        # plot.plot_weights_SSConv(dW, 'dW SSConv', (1000,600), (400, 400), True, 1, device)

        #Computing update
        W_new = state.weights + dW

        #Computing weights in range from zero to one
        Wmin = torch.amin(W_new, dim = (1,2,3,4)).view(state.weights.shape[0], 1, 1, 1, 1).expand(-1, *state.weights.shape[1:])
        Wmax = torch.amax(W_new, dim = (1,2,3,4)).view(state.weights.shape[0], 1, 1, 1, 1).expand(-1, *state.weights.shape[1:])
        W_mean = (W_new - Wmin)/(Wmax - Wmin)
        
        #Set up current Li equal to moving average of previous Li 
        Li_cur = torch.mean(Li_buffer_new, dim = 0).clone().unsqueeze(0).expand(len(spike_indices[0]), -1, -1)
        #Assign new Li values to corresponding maps
        Li_cur[tuple(torch.arange(len(spike_indices[0]))), spike_indices[0], spike_indices[1]] = 1/(shape[1]*shape[2]*shape[3]**2)*torch.sum((X-W_mean[spike_indices[1]])**2, dim = (1, 2,3,4))
        #Average Li-values per map 
        Li_cur = torch.mean(Li_cur, dim = 0)


        #Shifting entries in buffer 
        Li_buffer_old = Li_buffer_new[:-1].clone()
        Li_buffer_new[1:] = Li_buffer_old
        Li_buffer_new[0] = Li_cur

        #Updating number of spikes per neuron 
        nu_spikes_new[tuple(spike_indices[0]), tuple(spike_indices[1]), tuple(spike_indices[3]), tuple(spike_indices[4])] += 1


    #print ("Learning took ", time.time() - start_time, "to run")

    return STDPFedericoState(W_new, Li_buffer_new, nu_spikes_new), X







