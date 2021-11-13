
import torch 

#Image dimensions
height = 180 
width = 240 

#Number of batches
N = 1

#Number of channels (on/off)
ch = 2

#Number of time step 
t_ver = height
t_hor = width

#Thickness of bar in pixels 
th = 30 

#Set up tuples for vertical and horizontal range 
range_ver = torch.arange(height - th -1, -1, -1)
range_hor = torch.arange(width - th)

range_t_ver = torch.arange(t_ver - th)
range_t_hor = torch.arange(t_hor- th)

#Set up tensors to collect data
input_ver = torch.zeros((t_ver, N, ch, height, width))
input_hor = torch.zeros((t_hor, N, ch, height, width))

OF_ver = torch.zeros((t_ver, height, width, 2))
OF_hor = torch.zeros((t_hor, height, width, 2))


#All off events 
input_ver[tuple(range_t_ver), 0, 1, tuple(range_ver), :] = 1
input_ver[tuple(range_t_ver), 0, 0, tuple(range_ver+ th), :] = 1


input_hor[tuple(range_t_hor), 0, 0, :, tuple(range_hor)] = 1
input_hor[tuple(range_t_hor), 0, 1, :, tuple(range_hor + th)] = 1

#Compute OF tensors 
OF_ver[tuple(range_t_ver), tuple(range_ver), :] = torch.tensor([0, 1.])
OF_ver[tuple(range_t_ver), tuple(range_ver+ th), :] = torch.tensor([0, 1.])

OF_hor[tuple(range_t_hor),:, tuple(range_hor)] = torch.tensor([1., 0])
OF_hor[tuple(range_t_hor), :, tuple(range_hor+ th)] = torch.tensor([1., 0])

torch.save(input_ver, 'data_tensors/lines/input_ver_art.pt')
torch.save(input_hor, 'data_tensors/lines/input_hor_art.pt')

torch.save(OF_ver, 'data_tensors/ground_truths/lines/input_ver_art.pt')
torch.save(OF_hor, 'data_tensors/ground_truths/lines/input_hor_art.pt')
