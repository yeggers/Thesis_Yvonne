import bz2
import pickle
from inspect import currentframe

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.optimize import curve_fit

# #Define array containing all threshold values
v_th_values_rot_disk = np.array([0.1, 0.3, 0.5, 0.7,  0.9, 1.1, 1.3, 1.5])

#v_th_values_rot_disk = np.array([0.5])

cutoff = 70000

#Define length of moving average window 
w_rot_disk = 30


#Set up array to collect results for different voltage thresholds 
overlap_buffers_rot_disk = np.zeros((len(v_th_values_rot_disk), cutoff - w_rot_disk + 1))
overlap_buffers_rot_disk_a = np.zeros((len(v_th_values_rot_disk), cutoff - w_rot_disk + 1))


colours = ['k', 'r', 'b', 'c', 'm', 'g', 'y', 'g']


#Define exponential function for curve fitting 
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

#Define 3D exponential function for curvefitting
def func3D(data, a, b, c, d):
    return np.array(a*np.exp(-b*data[0]-c*data[1])+d)




#Loop through all threshold values 
for i in range(len(v_th_values_rot_disk)):
   
    #Define filename containing stiffness history
    filename_rot_disk_vth = 'pickles/rot_disk/stiffness/vth/overlap_alpha_0.2_v_th_{threshold}_lambda_X_0.005_lambda_v_0.005.pickle'.format(threshold = v_th_values_rot_disk[i])

    #Write data into tensor
    with open(filename_rot_disk_vth, 'rb') as f:
        overlap_buffer_rot_disk = pickle.load(f)[:cutoff]
    x_rot_disk = np.arange(len(overlap_buffer_rot_disk))

    # #Use this after performing new runs (pickles are now in compressed format)
    # filename_rot_disk_vth = 'pickles/rot_disk/stiffness/vth/stiffness_alpha_0.2_vth_{threshold}_lambda_X_0.005_lambda_v_0.005_iterations_20.pbz2'.format(threshold = v_th_values_rot_disk[i])
   
    # #Write data into tensor
    # data = bz2.BZ2File(filename_rot_disk_vth, 'rb')
    # overlap_buffer_rot_disk = cPickle.load(data)
    # x_rot_disk = np.arange(len(overlap_buffer_rot_disk))
    
    #Replace NaN-values (no spikes occured in the map) with zero values, no longer needed after performing new runs. Stiffness value of -1 is now assigned when no spikes occur
    overlap_buffer_rot_disk[torch.where(overlap_buffer_rot_disk != overlap_buffer_rot_disk)] = 0
   
  
    

    #Apply moving average to the array
    mv_avg_rot_disk  = np.convolve(np.array(overlap_buffer_rot_disk.cpu()), np.ones(w_rot_disk), 'valid')/w_rot_disk
   
    #Append overlap buffer to 
    #overlap_buffers_rot_disk[i] = mv_avg_rot_disk
    #overlap_buffers_rot_disk_a[i] = mv_avg_rot_disk_a

    #Apply exponential curve fitting
    # TODO: fix overflow warning during curve fit
    #To raw data
    popt_rot_disk, pcov_rot_disk = curve_fit(func, x_rot_disk, overlap_buffer_rot_disk)
    # #To moving average
    # x_rot_disk = np.arange(len(mv_avg_rot_disk))
    # popt_rot_disk, pcov_rot_disk = curve_fit(func, x_rot_disk, mv_avg_rot_disk)

    

    #Create figure 
    plt.figure('stiffnesses', figsize=(18.5, 30))
    #Plot stiffness values 
    plt.plot(mv_avg_rot_disk, label = 'v_th = {threshold}, filter length = {filter_length} ms'.format(threshold = v_th_values_rot_disk[i], filter_length = w_rot_disk), linestyle = '--', color = colours[i])
    #Plot curve fits 
    plt.plot(x_rot_disk, func(x_rot_disk, *popt_rot_disk), color = colours[i], label = 'v_th = {threshold} (exponential fit)'.format(threshold = v_th_values_rot_disk[i]))
    plt.ylabel('Stiffness [-]')
    plt.xlabel('Time step [ms]')
    #plt.ylim((-0.2,1.2))
    #plt.xlim((0,45000))
    plt.grid()


plt.legend()
plt.show()
