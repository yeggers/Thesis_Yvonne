import torch 
from models.SNN_Federico2 import SNN_Federico
from SNN_param import SNN_param
import tools.plotting as plot


'''This script can be used to train the MSConv layer of the SNN presented in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception" using the STDP rule presented there'''


#Simulation timestep 
dt = 10**(-3)

#Batch size 
batch_size = 1

#Specify which parameters to use ('checkerboard' for checkerboard data, 'rot_disk' for rotating disk data and 'roadmap' for roadmap data)
par_name = 'rot_disk'
par = SNN_param(par_name).define_data()

#Specify which weights to use 
SSConv_weights = 'weights/SSConv/rot_disk/SSConvWeights_rot_disk_alpha02_lambda005_vth5.pt'
MSConv_weights_exc = 'weights/MSConv/rot_disk/MSConvWeights_rot_disk_exc_test3.pt'
MSConv_weights_inh = 'weights/MSConv/rot_disk/MSConvWeights_rot_dsik_inh_test3.pt'



#Checking if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")

else:
    device = torch.device("cpu")
    print("Running on the CPU")

#Loading input data
data = torch.load('data_tensors/rotating_disk/use/rot_disk.pt').to(device)

  
#Performing inference
seq_length = data.shape[0]
SNN = SNN_Federico(SSConv_weights, MSConv_weights_exc, MSConv_weights_inh, par, device, batch_size, dt = 10**(-3))

for ts in range(seq_length):
    z = SNN.forward(data[ts])









