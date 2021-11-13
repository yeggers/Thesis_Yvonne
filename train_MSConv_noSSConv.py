from models.training_MSConv_noSSConv import TrainMSConv_noSSConv
import torch 
from SNN_param_noSSConv import SNN_param_noSSConv

'''This script can be used to train the MSConv layer of the SNN presented in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception" using the STDP rule presented there'''


def main_MSConv_noSSConv(dt, batch_size, iterations, new_weights, par_name, par, weights_name_exc, weights_name_inh, device):
    """Perform training of MSConv layer without SSConv layer
    
    Parameters:
        dt (float): simulation time step 
        batch_size (int): batch size 
        iterations (int): number of sequences used for training
        new_weights (bool): switch specifying whether or not new weights shall be created 
        par_name(string): data used for training
        par (SNN_param): parameters of the SNN
        name_weights_exc (string): name of the excitatory weights 
        name_weights_ing (string): name of the inhibitory weights
        device (torch.device): device on which computations are performed
    """
   
    #Specify desired name of initial excitatory weights or name of the existing weights 
    weights_name_exc = 'weights/MSConv/{par}/noSSConv/{weights_name}'.format(par = par_name, weights_name = weights_name_exc)

    #Specify desired name of initial excitatory weights or name of the existing weights 
    weights_name_inh = 'weights/MSConv/{par}/noSSConv/{weights_name}'.format(par = par_name, weights_name = weights_name_inh)

    #Initialize new weights if required 
    if new_weights:
        weights_exc = par.MSConv_w_init_ex*torch.ones(*par.MSConv_weight_shape)
        torch.save(weights_exc, weights_name_exc)
        
        weights_inh = torch.zeros(*par.MSConv_weight_shape)
        torch.save(weights_inh, weights_name_inh)

    #Defining directory of input data
    directory = par.directory
        
    #Performing training
    train = TrainMSConv_noSSConv(device, par = par, dt = dt, MSConvWeights_exc = weights_name_exc, MSConvWeights_inh = weights_name_inh) 
    s_STDP_exc, s_STDP_inh = train.forward(dir = directory, batch_size = batch_size, iterations = iterations, device = device)

    #Saving results
    torch.save(s_STDP_exc.weights, weights_name_exc)
    torch.save(s_STDP_inh.weights,  weights_name_inh)

    return s_STDP_exc, s_STDP_inh

if __name__ == "__main__":
    
    #Simulation timestep 
    dt = 10**(-3)

    #Batch size (Not that only one sequenze is available for the rotating disk and the checkerbord)
    batch_size = 1

    #Number of iterations to perform during training
    iterations = 1

    #Specify whether to initialize new weights or to use existing ones 
    new_weights = False

    #Specify which parameters to use ('checkerboard' for checkerboard data, 'rot_disk' for rotating disk data and 'roadmap' for roadmap data)
    par_name = 'rot_disk'
    par = SNN_param_noSSConv(par_name).define_data()

    #Name of weights 
    weights_name_exc = 'MSConvWeights_rot_disk_exc_test000.pt'
    weights_name_inh = 'MSConvWeights_rot_disk_inh_test000.pt'

    # #Name of weights 
    # weights_name_exc = 'MSConvWeights_rot_disk_exc_test3.pt'
    # weights_name_inh = 'MSConvWeights_rot_dsik_inh_test3.pt'
    
    #Checking if GPU is available
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        print("Running on the GPU")

    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    s_STDP_exc, s_STDP_inh = main_MSConv_noSSConv(dt, batch_size, iterations, new_weights, par_name, par, weights_name_exc, weights_name_inh, device)







