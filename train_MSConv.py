from models.training_MSConv import TrainMSConv
import torch 
from SNN_param import SNN_param
from datetime import datetime

'''This script can be used to train the MSConv layer of the SNN presented in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception" using the STDP rule presented there'''


#Simulation timestep 
dt = 10**(-3)

#Batch size (Not that only one sequenze is available for the rotating disk and the checkerbord)
batch_size = 1

#Number of iterations to perform during training
iterations = 10

#Specify whether to initialize new weights or to use existing ones 
new_weights = False

#Specify whether training should be performed 
training = False

#Specify which parameters to use ('checkerboard' for checkerboard data, 'rot_disk' for rotating disk data and 'roadmap' for roadmap data)
par_name = 'rot_disk'
par = SNN_param(par_name).define_data()

#Specify directory of trained SSConv weights 
if par_name == 'rot_disk':
    #Rotating disk
    weights_SSConv = 'weights/SSConv/rot_disk/SSConvWeights_rot_disk_Fede_32.pt'

elif par_name == 'roadmap':
    #Roadmap
    weights_SSConv = 'weights/SSConv/roadmap/SSConvWeights_roadmap_Fede.pt'


#Initialize new weights if required 
if new_weights:
    weights_name_exc = 'MSConvWeights_exc_alpha_{alpha}_vth_{vth}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_t_{time}'.format(alpha = par.MSConv_alpha, vth = par.MSConv_v_th, lambda_X = par.MSConv_lambda_X, lambda_v = par.MSConv_lambda_v, time = datetime.now())
    weights_name_exc = 'weights/MSConv/{data}/{weights_name}.pt'.format(data = par_name, weights_name = weights_name_exc)

    weights_name_inh = 'MSConvWeights_inh_alpha_{alpha}_vth_{vth}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_t_{time}'.format(alpha = par.MSConv_alpha, vth = par.MSConv_v_th, lambda_X = par.MSConv_lambda_X, lambda_v = par.MSConv_lambda_v, time = datetime.now())
    weights_name_inh = 'weights/MSConv/{data}/{weights_name}.pt'.format(data = par_name, weights_name = weights_name_inh)
    
    weights_exc = par.MSConv_w_init_ex*torch.ones(*par.MSConv_weight_shape)
    torch.save(weights_exc, weights_name_exc)
    
    weights_inh = torch.zeros(*par.MSConv_weight_shape)
    torch.save(weights_inh, weights_name_inh)

else:
    if par_name == 'rot_disk':
        #Rotating disk 
        weights_name_exc = 'MSConvWeights_rot_disk_exc_Fede'
        weights_name_inh = 'MSConvWeights_rot_disk_inh_Fede'
    
    elif par_name == 'roadmap':
        # #Roadmap
        weights_name_exc = 'MSConvWeights_roadmap_exc_Fede'
        weights_name_inh = 'MSConvWeights_roadmap_inh_Fede'

    # weights_name_inh = 'MSConvWeights_inh_alpha_0.1_vth_1_lambda_X_0.03_lambda_v_0.03_t_2021-11-12 09:17:02.359518'
    # weights_name_exc = 'MSConvWeights_exc_alpha_0.1_vth_1_lambda_X_0.03_lambda_v_0.03_t_2021-11-12 09:17:02.359498'

    # weights_name_inh = 'MSConvWeights_inh_alpha_0.1_vth_1_lambda_X_0.03_lambda_v_0.03_t_2021-11-12 10:14:44.464060'
    # weights_name_exc = 'MSConvWeights_exc_alpha_0.1_vth_1_lambda_X_0.03_lambda_v_0.03_t_2021-11-12 10:14:44.464040'


    #weights_name = 'SSConvWeights_alpha_0.01_vth_0.7_lambda_X_0.007_lambda_v_0.007_t_2021-11-11 14:22:48.797588'
    weights_name_exc = 'weights/MSConv/{data}/{weights_name}.pt'.format(data = par_name, weights_name = weights_name_exc)
    weights_name_inh = 'weights/MSConv/{data}/{weights_name}.pt'.format(data = par_name, weights_name = weights_name_inh)

    



#Checking if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")

else:
    device = torch.device("cpu")
    print("Running on the CPU")


  
#Performing training
train = TrainMSConv(device, par = par, weights_name_exc = weights_name_exc, weights_name_inh = weights_name_inh,  dt = dt, SSConvWeights = weights_SSConv, MSConvWeights_exc = weights_name_exc, MSConvWeights_inh = weights_name_inh) 









s_STDP_exc, s_STDP_inh = train.forward(dir = par.directory, batch_size = batch_size, iterations = iterations, training = training,  device = device)

#Saving results
torch.save(s_STDP_exc.weights, weights_name_exc)
torch.save(s_STDP_inh.weights,  weights_name_inh)







