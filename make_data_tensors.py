import torch 
import tools.importing_dvs
import bz2
import pickle
import _pickle as cPickle
import torchvision
import pandas as pd
from SNN_param import SNN_param



#Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Running on the GPU")
    print()
    print()

else:
    device = torch.device("cpu")
    print("Running on the CPU")
    print()
    print()

#Make Fede's weights, resulting tensors are transposed since x- and y-axes are flipped in Federico's weights
#Rotating disk_SSConv
par_name = 'rot_disk'
par = SNN_param(par_name).define_data()
data = torch.tensor(pd.read_csv('weights/SSConv/rot_disk/layer_0_excweights.csv', header = None, usecols=[0]).values).view(par.SSConv_weight_shape)
data = torch.transpose(data, 3,4)
torch.save(data, 'weights/SSConv/rot_disk/SSConvWeights_rot_disk_Fede_32.pt')

#Rotating disk MSConv
data = torch.tensor(pd.read_csv('weights/MSConv/rot_disk/layer_2_excweights.csv', header = None, usecols=[0]).values).view(*par.MSConv_weight_shape[0:2], *par.MSConv_weight_shape[3:5], par.MSConv_weight_shape[2])
data = data.permute(0, 1, 4, 3, 2)
torch.save(data, 'weights/MSConv/rot_disk/MSConvWeights_rot_disk_exc_Fede.pt')

data = torch.tensor(pd.read_csv('weights/MSConv/rot_disk/layer_2_inhweights.csv', header = None, usecols=[0]).values).view(*par.MSConv_weight_shape[0:2], *par.MSConv_weight_shape[3:5], par.MSConv_weight_shape[2])
data = data.permute(0, 1, 4, 3, 2)
torch.save(data, 'weights/MSConv/rot_disk/MSConvWeights_rot_disk_inh_Fede.pt')

#Roadmap_SSConv
par_name = 'roadmap'
par = SNN_param(par_name).define_data()
data = torch.tensor(pd.read_csv('weights/SSConv/roadmap/layer_0_excweights.csv', header = None, usecols=[0]).values).view(par.SSConv_weight_shape)
data = torch.transpose(data, 3,4)
torch.save(data, 'weights/SSConv/roadmap/SSConvWeights_roadmap_Fede.pt')

#Rotating disk MSConv
data = torch.tensor(pd.read_csv('weights/MSConv/roadmap/layer_2_excweights.csv', header = None, usecols=[0]).values).view(*par.MSConv_weight_shape[0:2], *par.MSConv_weight_shape[3:5], par.MSConv_weight_shape[2])
data = data.permute(0, 1, 4, 3, 2)
torch.save(data, 'weights/MSConv/roadmap/MSConvWeights_roadmap_exc_Fede.pt')

data = torch.tensor(pd.read_csv('weights/MSConv/roadmap/layer_2_inhweights.csv', header = None, usecols=[0]).values).view(*par.MSConv_weight_shape[0:2], *par.MSConv_weight_shape[3:5], par.MSConv_weight_shape[2])
data = data.permute(0, 1, 4, 3, 2)
torch.save(data, 'weights/MSConv/roadmap/MSConvWeights_roadmap_inh_Fede.pt')



#Make rotating disk tensor
print("Making rotating disk data")
tools.importing_dvs.import_DVS(180, 240).make_rotating_disk_data('data/disk/IMU_rotDisk/events.csv', 'data_tensors/rotating_disk/use')

#Make ODA tensors
print()
print("Making ODA data")
tools.importing_dvs.import_DVS(180, 240).make_ODA_data('data/ODA_Dataset/dataset', 'data_tensors/ODA_dataset')

# #Make checkerboard tensors. Aedat file for checkerboard data not included since it is too big for github
# print()
# print("Making checkerboard tensors")
# tools.importing_dvs.import_DVS(180, 240).make_checkerboard_data('data/checkerboard', 'data_tensors/checkerboard')

#Make roadmap tensors
print()
print("Making roadmap data")
tools.importing_dvs.import_DVS(264, 320).make_roadmap_data('data/roadmap', 'data_tensors/roadmap')








