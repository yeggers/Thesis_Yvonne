from models.training_SSConv import TrainSSConv
import torch 
from SNN_param import SNN_param
from datetime import datetime 
import torch


#Simulation timestep 
dt = 10**(-3)

#Batch size (Only one sequence is available for the rotating disk)
batch_size = 1

#Number of iterations to perform during training
iterations = 1

#Specify whether to initialize new weights or to use existing ones 
new_weights = False

#Specify whether training should be performed 
training = False

#Specify which parameters to use ('checkerboard' for checkerboard data, 'rot_disk' f
# or rotating disk data and 'roadmap' for roadmap data)
par_name = 'rot_disk'
par = SNN_param(par_name).define_data()


#Initialize new weights if required 
if new_weights:
    weights_name = 'SSConvWeights_alpha_{alpha}_vth_{vth}_lambda_X_{lambda_X}_lambda_v_{lambda_v}_t_{time}'.format(alpha = par.SSConv_alpha, vth = par.SSConv_v_th, lambda_X = par.SSConv_lambda_X, lambda_v = par.SSConv_lambda_v, time = datetime.now())
    weights_name = 'weights/SSConv/{data}/{weights_name}.pt'.format(data = par_name, weights_name = weights_name)
    weights = par.SSConv_w_init*torch.ones(*par.SSConv_weight_shape)
    torch.save(weights, weights_name)
    

else:
    if par_name == 'rot_disk':
        #Rotating disk 
        weights_name = 'SSConvWeights_rot_disk_Fede_32'
    elif par_name == 'roadmap':
        #Roadmap
        weights_name = 'SSConvWeights_roadmap_Fede'
   
    weights_name = 'weights/SSConv/{data}/{weights_name}.pt'.format(data = par_name, weights_name = weights_name)


#Checking if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")

else:
    device = torch.device("cpu")
    print("Running on the CPU")


#Performing training
train = TrainSSConv(device, par, weights_name, dt = dt, SSConv_weights= weights_name) 
s_STDP = train.forward(dir = par.directory, batch_size = batch_size, iterations = iterations, training = training, device = device)

#Saving result
torch.save(s_STDP.weights, weights_name)


