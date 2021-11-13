import torch 
import importing_dvs as i
import matplotlib.pyplot as plt 
import numpy as np


# height = 180 
# width = 240 
# ch = 2

# dir_event = '/home/yvonne/Documents/uni_stuff/thesis/code/Thesis_SNN/data/checkerboard/checkerboard.aedat4'
# DVS_data = i.import_DVS(dir_event, height = height, width = width) #264, 320 #180, 240
# data = DVS_data.read_aedat_data() 


# hist_hor = torch.zeros(ch, width)
# hist_ver = torch.zeros(ch, height)


# for event in data: 
#     hist_hor[int(event[3]), event[1]] += 1
#     hist_ver[int(event[3]), event[2]] += 1

# #Plotting histograms 

# #horizontal off events 
# plt.figure()
# plt.subplot(2,2,1)
# plt.title("Off-events in horizontal direction")
# plt.bar(np.arange(width), hist_hor[0])
# plt.ylabel("Number of off-events within sequence")
# plt.xlabel("x-pixel")
# plt.grid(True)



# #horizontal on events 
# plt.subplot(2,2,2)
# plt.title("On-events in horizontal direction")
# plt.bar(np.arange(width), hist_hor[1])
# plt.ylabel("Number of on-events within sequence")
# plt.xlabel("x-pixel")
# plt.grid(True)


# #vertical off events 
# plt.subplot(2,2,3)
# plt.title("Off-events in vertical direction")
# plt.bar(np.arange(height), hist_ver[0])
# plt.ylabel("Number of off-events within sequence")
# plt.xlabel("y-pixel")
# plt.grid(True)



# #vertical on events 
# plt.subplot(2,2,4)
# plt.title("On-events in vertical direction")
# plt.bar(np.arange(height), hist_ver[1])
# plt.ylabel("Number of on-events within sequence")
# plt.xlabel("y-pixel")
# plt.grid(True)
# plt.show()



height = 264
width = 320
ch = 2
#dir_event = '/home/yvonne/Documents/uni_stuff/thesis/code/Thesis_SNN/data/disk/IMU_rotDisk/events.csv'
dir_event = '/home/yvonne/Documents/uni_stuff/thesis/code/cuSNN/cuSNN-samples/data/roadmap/final/final_0/events.csv'
DVS_data = i.import_DVS(height = height, width = width)
data = DVS_data.read_csv_data(dir_event, fix_stripes = True)

hist_hor = torch.zeros(ch, width)
hist_ver = torch.zeros(ch, height)
hist = torch.zeros(height, width)


for event in range(len(data)):
    hist_hor[int(data.pol[event]), int(data.x[event])] += 1
    hist_ver[int(data.pol[event]), int(data.y[event])] += 1
    hist[int(data.y[event]), int(data.x[event])] += 1



#horizontal off events 
plt.figure()
plt.subplot(2,2,1)
plt.title("Off-events in horizontal direction")
plt.bar(np.arange(width), hist_hor[0])
plt.ylabel("Number of off-events within sequence")
plt.xlabel("x-pixel")
plt.grid(True)



#horizontal on events 
plt.subplot(2,2,2)
plt.title("On-events in horizontal direction")
plt.bar(np.arange(width), hist_hor[1])
plt.ylabel("Number of on-events within sequence")
plt.xlabel("x-pixel")
plt.grid(True)


#vertical off events 
plt.subplot(2,2,3)
plt.title("Off-events in vertical direction")
plt.bar(np.arange(height), hist_ver[0])
plt.ylabel("Number of off-events within sequence")
plt.xlabel("y-pixel")
plt.grid(True)



#vertical on events 
plt.subplot(2,2,4)
plt.title("On-events in vertical direction")
plt.bar(np.arange(height), hist_ver[1])
plt.ylabel("Number of on-events within sequence")
plt.xlabel("y-pixel")
plt.grid(True)
plt.show()









    


 