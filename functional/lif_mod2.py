

import math
import time
from collections import deque
from typing import NamedTuple, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tools.plotting as plot
import torch
import torchvision
from norse.torch.functional.lif import (LIFFeedForwardState,
                                        lif_feed_forward_step, lif_step)
from norse.torch.functional.lif_refrac import (LIFRefracFeedForwardState,
                                               LIFRefracParameters,
                                               LIFRefracState,
                                               compute_refractory_update,
                                               lif_feed_forward_step, lif_step)
from norse.torch.functional.threshold import threshold



class LIFModState(NamedTuple):
    """State of a LIFRefrac neuron with multisynaptic connections and delays applied to the output.

    Parameters:
        lifRefrac (LIFRefracState): state of the LIFRefrac neuron integration
        buffer (torch.Tensor): buffer containing time delays of output
        stiffness_buffer (torch.Tensor): buffer containing previous values of stiffness for average filtering
    """

    lifRefrac: LIFRefracState
    buffer: torch.Tensor
    stiffness_buffer: torch.Tensor



class LIFModStateNeuronTrace(NamedTuple):
    """State of a modified LIF neuron with neuron trace.

    Parameters:
        lifMod (LIFModState): state of the LIFMod neuron integration
        X (torch.Tensor): tensor containing neuron trace of output spikes 
      
    """

    lifMod:  LIFModState
    X: torch.Tensor 
  

class LIFModParameters(NamedTuple):
    """Parameters of a LIFRefrac neuron with multisynaptic connections and delays applied to the output.

    Parameters:
        lifRefrac (LIFRefracParameters): parameters of the LIFRefrac neuron integration
        delays (torch.Tensor): delays applied to the output of the neuron, one delay per synaptic connection [ms]
        m (torch.Tensor): number of synapses per neuron 
        
        lambda_vth (torch.Tensor): time constant for the adaptive voltage threshold
        alpha_vth (torch.Tensor): scaling factor for adaptive voltage threshold
        vth_rest (torch.Tensor): resting voltage threshold
        vth_conv_parmas (torch.Tensor): convolutional parameters defining window size for voltage threshold update
        stiffness_goal_inf (torch.Tensor): target for the stiffness value during inference
        target_parameters (torch.tensor): parameters of exponential function describing the target stiffness curve
        len_stiffness_buffer (torch.tensor): length of buffer for average filtering of stiffness
        v_th_gains (torch.tensor): gains for adaptive voltage threshold
        
        tau_max_gain (torch.tensor): gain for adaptive tau_max 
    """

    #Modfied LIF neuron paramters
    lifRefrac: LIFRefracParameters = LIFRefracParameters()
    delays: torch.Tensor = torch.Tensor([1]).long()
    m: torch.Tensor = torch.as_tensor(1) 
    
    #Parameters for adaptive threshold 
    lambda_vth: torch.Tensor = torch.as_tensor(0.005)
    alpha_vth: torch.Tensor = torch.as_tensor(0.01)
    vth_rest: torch.Tensor = torch.as_tensor(0)
    vth_conv_params: torch.Tensor = torch.Tensor([5,1,2])
    stiffness_goal_inf: torch.Tensor = torch.as_tensor(1)
    target_parameters: list = [12.327922591384661, 7.516300463119961e-05, 1.796295780751845], 
    len_stiffness_buffer: torch.Tensor = torch.as_tensor(20), 
    v_th_gains: torch.Tensor = torch.Tensor([0.001, 0.01, 0.000001]), 
    
    #Paramter for adaptive maximum delay
    tau_max_gain:  torch.Tensor = torch.as_tensor(1.5)


class LIFModParametersNeuronTrace(NamedTuple):
    """Parameters of a modified LIF neuron with neuron trace.

    Parameters:
        lifMod (LIFModParameters): parameters of the LIFMod neuron integration
        alpha_mod (torch.Tensor): scaling factor of neuron trace  
        lambda_x (torch.Tensor) : time constant of decay in neuron trace
    """

    lifMod: LIFModParameters = LIFModParameters()
    alpha_mod: torch.Tensor = torch.as_tensor(0.25)
    lambda_x: torch.Tensor = torch.as_tensor(5)
    



def compute_presynaptic_trace_update(
    z_new: torch.Tensor,
    state: LIFModStateNeuronTrace, 
    p: LIFModParametersNeuronTrace = LIFModParametersNeuronTrace(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor]:
     
    """Compute the neuron trace update according to the specified ODE: 
     .. math::
        \begin{align*}
            \dot{X} &= 1/\lambda_{\text{x}} (-X + \alpha z_new)\\
        \end{align*}
    
    Parameters
        z_new (torch.Tensor): output spikes of neural layer.
        state (LIFModStateNeuronTrace): initial state of the modified LIF neuron with neuron trace.
        p (torch.Tensor): paramters of the modified LIF neuron with neuron trace.
        dt (float): simulation time step 
    """

    #Compute presynaptic trace update 
    dX = dt*1/p.lambda_x *(-state.X + p.alpha_mod*z_new)
    X = state.X + dX

    return X



def compute_wta(
    z: torch.Tensor,
    X: torch.Tensor,
    rho_new: torch.Tensor, 
    v_decayed: torch.Tensor,
    v_new: torch.Tensor, 
    p: LIFModParameters, 
    k: tuple, 
    s: tuple, 
    training, 
    device
)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """Perform WTA on output spikes.
    
    Parameters
        z_new (torch.Tensor): output spikes
        X (torch.Tensor): presynaptic trace
        rho_new (torch.Tensor): refractory state
        v_decayed (torch.Tensor): voltages of neurons before they spikes
        v_new (torch.Tensor): voltages of neurons after they spiked
        p (LIFModParameters): parameters of the modified LIF neurons
        k (tuple): convolutional kernel size of the map for which WTA is applied
        s (tuple): convolutional stride of the map for which WTA is applied
        training (bool): flag indicating whether map is being trained
        device (torch.device): device on which computations shall be performed
    """
    
    #start_time = time.time()
    
    #Extracting spike indices 
    spike_indices = torch.vstack(torch.where(z == 1))
  
    # # Shuffling indices to prevent one layer from getting all spikes during first iteration 
    # spike_indices = spike_indices[:, torch.randperm(spike_indices.size()[1])]

    #Defining boxes describing presynaptic traces
    boxl = s[1:]*torch.transpose(spike_indices[(3,2), :], 0, 1).float()
    boxu = (s[1:]*torch.transpose(spike_indices[(3,2), :], 0, 1) + k[1:]).float()

    # #Defining boxes describing direct neural neighbourhood of each neuron
    # boxl = (torch.transpose(spike_indices[(3,2), :], 0, 1) - 1).float()
    # boxu = (torch.transpose(spike_indices[(3,2), :], 0, 1) + 2).float()

    #Creating tensor specifying to which batch each box belongs
    idxs_batch = spike_indices[0, :]
    
    #Creating tensor specifying to which map each box belongs
    idxs_map = spike_indices[1,:]

    #Performing WTA across all maps during training and only within the maps otherwise
    
    #only differentiate between different batches (= competition across maps)
    idxs_batch = idxs_batch
    
    #differentiate between batches and different maps (= no competition across maps)
    idxs_maps = idxs_batch *z.shape[1] + idxs_map

    scores = v_decayed[tuple(spike_indices)]

    #Performing WTA with competition across maps
    winds_batch = torchvision.ops.batched_nms(torch.cat((boxl, boxu), 1), scores, idxs_batch, 0) 
   
    #Performing WTA with no competition across maps
    winds_maps = torchvision.ops.batched_nms(torch.cat((boxl, boxu), 1), scores, idxs_maps, 0) 

    #Determine spike indices after duplicates within windows have been removed
    spike_indices_maps = spike_indices[:, tuple(winds_maps)]
    spike_indices_batch = spike_indices[:, tuple(winds_batch)]

    #Create outspikes for WTA with only maps and only batch 
    z_new_maps = torch.zeros_like(z)
    z_new_batch = torch.zeros_like(z)
    z_new_maps[tuple(spike_indices_maps)] = 1
    z_new_batch[tuple(spike_indices_batch)] = 1


    if training == False:
        winds = winds_maps
        z_new = z_new_maps
    else:
        winds = winds_batch
        z_new = z_new_batch

    #Determine winning boxes
    boxes = torch.cat((boxl, boxu),1)[tuple(winds), :]
    box_indices = spike_indices[:, winds]
 
    #Inhibit neurons in neighborhood of spiking neuron 
    dis = torch.floor((k - 1)/s).long()
    dis = torch.tensor([0,3,3]).long()

    for i in range(len(winds)):
        rho_new[spike_indices[0, winds[i]], : , spike_indices[2, winds[i]] - dis[1] : spike_indices[2, winds[i]] + dis[1] + 1, spike_indices[3, winds[i]] - dis[2]: spike_indices[3, winds[i]] + dis[2] + 1] = p.lifRefrac.rho_reset
        v_new[spike_indices[0, winds[i]], : , spike_indices[2, winds[i]] - dis[1] : spike_indices[2, winds[i]] + dis[1] + 1, spike_indices[3, winds[i]] - dis[2]: spike_indices[3, winds[i]] + dis[2] + 1] = 0
    

    #print ("WTA2 took ", time.time() - start_time, "to run")

    return z_new, z_new_batch, z_new_maps, boxes, box_indices



def delays(
    state: LIFModState, 
    p: LIFModParameters,
    z_new: torch.tensor, 
    dt: float
)-> Tuple[torch.Tensor]:
    
    """This function adds the latest spikes (z_new) to the buffer containing the delayed versions of the moodified LIF neuron spikes and removes the oldest ones.

    Parameters:
        state (LIFModState): state of the modified LIF neuron
        p (LIFModParameters): parameters of the modifed LIF neuron
        z_new (torch.Tensor): new output spikes 
        dt (float): simulation time step
    """
    
    #Moving all entries by one index 
    state.buffer.rotate(1) 
    
    #Putting latest spikes into buffer 
    state.buffer[0] = z_new
 
    #Computing factor of timestep [specified by dt] and delay[ms]
    factor = int(10**(-3)/dt)
    
    #Outputting spikes with desired delays 
    z_new = torch.stack(tuple(state.buffer))[factor*p.delays]
    
    #Reshaping the tensor to the suitable format for 3D convolution (Batch size, number of channels, number of synapses, height, width)
    z_new = z_new.permute(1,2,0,3,4)
    
    
    return z_new


def compute_tau_max_update(
    weights_exc: torch.tensor, 
    weights_inh: torch.tensor,
    map_indices: torch.tensor, 
    STD_goal: torch.tensor,
    p: LIFModParameters,
)-> Tuple[torch.Tensor]:

    """Compute tau_max update

    Parameters:
        weights_exc (torch.Tensor): Trained excitatory MSConv weights
        weights_inh (torch.Tensor): Trained inhibitory MSConv weights
        map_indices (torch.Tensor): Indices of maps in which spikes have occured
        STD_goal (torch.Tensor): Target value for STD
        p (LIFModParameters): parameters of the modifed LIF neuron 
    """

    #Compute weights
    weights = weights_exc + p.MSConv_beta * weights_inh

    #Compute mean of STD of total brightness per map across delays
    delay_weights_spikes = torch.sum(weights[map_indices] , dim = (1,3,4))
    delay_weights_spikes_STD = torch.mean(torch.std(delay_weights_spikes, dim = 1)/torch.mean(delay_weights_spikes, dim = 1))

    #Compute delay update 
    d_tau_max = (STD_goal - delay_weights_spikes_STD) *p.MsConv_tau_max *p.MSConv_tau_max_gain
    tau_max = p.MsConv_tau_max + d_tau_max

    return tau_max


           
def compute_vth_update(
    z_new_maps: torch.tensor,
    z_new_batch: torch.tensor,
    state: LIFModState,
    p: LIFModParameters,
    stiffness_goal: torch.tensor,
    dt: float, 
    device: torch.device
)-> Tuple[torch.Tensor]:
    
    """Compute voltage threshold update

    Parameters:
        z_new_maps (torch.Tensor): output spikes before WTA across maps 
        z_new_batch (torch.Tensor): output spikes after WTA across maps
        state (LIFModState): state of the modified LIF neuron
        p (LIFModParameters): parameters of the modifed LIF neuron 
        dt (float): simulation time step
        device (torch.device): device on which computations are performed
    """

    # #Compute number of maps in which spikes occur for every pixel location 
    # num_maps_maps = torch.sum(z_new_maps, dim = 1, keepdim = True).squeeze(2)
    # num_maps_batch = torch.sum(z_new_maps, dim = 1, keepdim = True).squeeze(2)

    #TODO: define the convolutional parameters in SNN_param.py in terms of output dimensions
    sum_maps = torch.nn.AvgPool3d(p.vth_conv_params[0], p.vth_conv_params[1], p.vth_conv_params[2])(z_new_maps)
    sum_batch = torch.nn.AvgPool3d(p.vth_conv_params[0], p.vth_conv_params[1], p.vth_conv_params[2])(z_new_batch)

    #Assign negative stiffness to all neurons
    stiffness = -1*torch.ones_like(sum_batch)
    
    #Check which spike indices remained after second WTA
    i_WTA = torch.where(sum_batch>0)

    #Compute stiffness for neurons in whose recpetive fields spikes occured
    stiffness[i_WTA] = (sum_maps[i_WTA] - sum_batch[i_WTA])/sum_batch[i_WTA]
 
    #Moving all entries by one index 
    moved = state.stiffness_buffer[0:-1].clone()
    state.stiffness_buffer[1:] = moved
    
    #Putting latest spikes into buffer 
    state.stiffness_buffer[0] = stiffness
 
    #Smoothness stiffness by averaging it over window of size buffer
    stiffness_ave = torch.mean(state.stiffness_buffer, dim = 0)

    #Parameters for plotting 3D-fitted curve
    # a_3D = 1.68690645e+01
    # b_3D = 3.73118612e-05
    # c_3D = 1.43743332e+00

    #Compute error between overlap goal and actual overlap
    error_stiffness = (stiffness_goal - stiffness_ave)
    
    ##Error derived from curve fit 
    #error_stiffness = (stiffness_goal - stiffness_ave_buffer)/(-c_3D*a_3D*np.exp(-b_3D*time - c_3D*p.SSConv_v_th))
   
    #Compute PID errors 
    #e_D = (e_P - error_stiffness)/dt
    e_P = error_stiffness
    #e_I += e_I

    #Compute voltage update
    #d_v_th = p.SSConv_v_th_training_gains *torch.array([e_P, e_I, e_D])*p.SSConv_v_th
    d_v_th = -p.v_th_gains[0]*e_P*p.lifRefrac.lif.v_th
    v_th = p.lifRefrac.lif.v_th + d_v_th
    #v_th = p.lifRefrac.lif.v_th 

    # #Compute average number of spiking maps within neural neighbourhood of neurons sharing presynaptic neurons
    # OL_maps_ave = torch.nn.AvgPool2d(p.vth_conv_params[0], p.vth_conv_params[1], p.vth_conv_params[2])(num_maps_maps)

    # #Compute voltage threshold update 
    # dv_th = dt * 1/p.lambda_vth * ((p.vth_rest- p.lifRefrac.lif.v_th) + p.alpha_vth*torch.gt(OL_maps_ave, 0)*(OL_maps_ave - p.N_ms))
    # v_th = p.lifRefrac.lif.v_th + dv_th
    # # print('dv_th: ', dv_th[torch.where(dv_th> 0)])
    # # print('v_th: ', v_th[torch.where(dv_th> 0)])
    
    
    return v_th, stiffness_ave

    




class LIFModFeedForwardStateNT(NamedTuple):
    """State of a modified feed forward LIF neuron with absolute refractory period.

    Parameters:
        lifRefrac (LIFRefracFeedForwardState): state of the feed forward LIFRefrac
                                   neuron integration
        X (torch.Tensor): presynaptic trace state 
        buffer (torch.Tensor): buffer containing delayed spiked
    """

    lifRefrac: LIFRefracFeedForwardState
    X: torch.Tensor
    buffer: torch.Tensor 


def lif_mod_feed_forward_step_NT(
    batch_size: torch.Tensor,
    i_new: torch.Tensor,
    state: LIFModFeedForwardStateNT,
    k: tuple, 
    s: tuple,
    device: torch.device,
    shape: tuple,
    p: LIFModParametersNeuronTrace,
    dt: float = 10**(-3),
) -> Tuple[torch.Tensor, torch.Tensor, LIFModFeedForwardStateNT]:
    
    """Computes a single euler-integration step of a modified feed forward
     LIF neuron-model with a refractory period. It takes as input the input current as generated by an arbitrary torch
    module or function and the sum of all the presynaptic traces for each neuron. More specifically it implements one integration
    step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}} - X
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes and X the sum of the neuron's presynaptic traces.

    Parameters:
        batch_size (torch.Tensor): batch size 
        i_new (torch.Tensor): input current 
        state (LIFModFeedForwardState): state at the current time step
        k (tuple): Kernel used for current layer 
        s(tuple): Stride used for current layer
        device (torch.device): device on which computations are performed 
        shape (tuple): shape of neuron layer (out_dim, height, width)
        p (LIFModParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """
  
    # compute voltage updatees
    #dv = dt * p.lifMod.lifRefrac.lif.tau_mem_inv * ((p.lifMod.lifRefrac.lif.v_leak - state.lifMod.lifRefrac.lif.v) + state.lifMod.lifRefrac.lif.i)
    dv = dt * p.lifMod.lifRefrac.lif.tau_mem_inv * ((p.lifMod.lifRefrac.lif.v_leak - state.lifRefrac.lif.v) + i_new.view(batch_size, *shape))
    v_decayed = state.lifRefrac.lif.v + dv

  
    # compute new spikes
    z_new = threshold(v_decayed - p.lifMod.lifRefrac.lif.v_th, p.lifMod.lifRefrac.lif.method, p.lifMod.lifRefrac.lif.alpha)
    
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lifMod.lifRefrac.lif.v_reset
   
    #compute refractory update
    v_new, z_new, rho_new = compute_refractory_update(state.lifRefrac, z_new, v_new, p.lifMod.lifRefrac) 

    #Applying delays 
    z_new = delays(state, p.lifMod, z_new, dt)
    
    #Compute presynaptic trace update 
    X_new = compute_presynaptic_trace_update(z_new, state, p, dt)
   
    return (
        z_new,
        LIFModFeedForwardStateNT(LIFRefracFeedForwardState(LIFFeedForwardState(v_new, i_new),rho_new), X_new, state.buffer),
    )



class LIFModFeedForwardStateWTA(NamedTuple):
    """State of a modified feed forward LIF neuron with absolute refractory period.

    Parameters:
        lifRefrac (LIFRefracFeedForwardState): state of the feed forward LIFRefrac
                                neuron integration
        buffer (torch.Tensor): buffer containing delayed spiked
        stiffness_bufffer (torch.Tensor): buffer containing previous values of stiffness

    """

    lifRefrac: LIFRefracFeedForwardState
    buffer: torch.Tensor 
    stiffness_buffer: torch.Tensor


def lif_mod_feed_forward_step_WTA(
    X: torch.Tensor,
    batch_size: torch.Tensor,
    i_new: torch.Tensor,
    state: LIFModFeedForwardStateWTA,
    k: tuple, 
    s: tuple,
    device: torch.device,
    training: bool, 
    t:float,
    shape: tuple,
    p: LIFModParameters,
    dt: float = 10**(-3),
) -> Tuple[torch.Tensor, torch.Tensor, LIFModFeedForwardStateWTA]:
    
    """Computes a single euler-integration step of a modified feed forward
     LIF neuron-model with a refractory period. It takes as input the input current as generated by an arbitrary torch
    module or function and the sum of all the presynaptic traces for each neuron. More specifically it implements one integration
    step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}} - X
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes and X the sum of the neuron's presynaptic traces.

    Parameters:
        X (torch.Tensor): presynaptic trace
        batch_size (torch.Tensor): batch size 
        i_new (torch.Tensor): input current 
        state (LIFModFeedForwardState): state at the current time step
        k (tuple): Kernel used for current layer 
        s(tuple): Stride used for current layer
        device (torch.device): device on which computations are performed 
        training (bool): indicates whether current SNN layer is being trained 
        t (int): current simulation time step
        shape (tuple): shape of neuron layer (out_dim, height, width)
        p (LIFModParameters): parameters of the lif neuron
        dt (float): Integration timestep to use
    """
  
    # compute voltage updatees
    #dv = dt * p.lifRefrac.lif.tau_mem_inv * ((p.lifRefrac.lif.v_leak - state.lifRefrac.lif.v) + state.lifRefrac.lif.i)
    dv = dt * p.lifRefrac.lif.tau_mem_inv * ((p.lifRefrac.lif.v_leak - state.lifRefrac.lif.v) + i_new.view(batch_size, *shape))
    v_decayed = state.lifRefrac.lif.v + dv

    # plot.plot_voltage_trace(v_decayed, 0, 0, 'voltage trace', (200, 200), (700, 500), 0, device)

    # plot.plot_voltage_trace(i_new.view(batch_size, *shape), 0, 0, 'current trace', (1000, 200), (700, 500), 1, device)

    # compute new spikes
    z_new = threshold(v_decayed - p.lifRefrac.lif.v_th, p.lifRefrac.lif.method, p.lifRefrac.lif.alpha)
    
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.lifRefrac.lif.v_reset

    #compute refractory update
    v_new, z_new, rho_new = compute_refractory_update(state.lifRefrac, z_new, v_new, p.lifRefrac) 

    #Apply WTA
    z_new, z_new_batch, z_new_maps, boxes, box_indices = compute_wta(z_new, X, rho_new, v_decayed, v_new, p,k, s, training, device)

    #Only apllying delays for layers that are not being trained (the delay is the delay for the nex layer)
    if training == False:
        
        #Applying delays 
        z_new = delays(state, p, z_new, dt)

        #Applying adaptive threshold 
        stiffness_goal = p.stiffness_goal_inf
        v_th, stiffness_ave = compute_vth_update(z_new_maps, z_new_batch, state, p, stiffness_goal, dt, device)
        #v_th, stiffness_ave = (p.lifRefrac.lif.v_th, 1)
      
       
    else: 
        z_new = z_new.unsqueeze(2)
       
        #Applying adaptive threshold 
        stiffness_goal = p.target_parameters[0]*np.exp(-p.target_parameters[1] *t) + p.target_parameters[2]
        v_th, stiffness_ave = compute_vth_update(z_new_maps, z_new_batch, state, p, stiffness_goal, t, device)
        #v_th, stiffness_ave = (p.lifRefrac.lif.v_th, 1)

   
    return (
        z_new,
        LIFModFeedForwardStateWTA(LIFRefracFeedForwardState(LIFFeedForwardState(v_new, i_new), rho_new), state.buffer, state.stiffness_buffer),
        v_th,
        np.array(boxes.cpu()).astype(int), 
        np.array(box_indices.cpu()).astype(int), 
        v_decayed,
        stiffness_ave, 
        stiffness_goal
    )


