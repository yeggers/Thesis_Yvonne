import matplotlib.pyplot as plt 
import cv2 
import numpy as np 
import torch
from typing import NamedTuple, Tuple
from itertools import product, combinations
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot2D_discrete(
    Input: torch.tensor, 
    height:int, 
    width: int, 
    batch_nu: int, 
    title: str,
    position: tuple,
    size: tuple, 
    waitKey: int, 
    dt: float
):
    '''This function plots the input data to the SSConv layer. 
    
    Parameters: 
        Input (torch.Tensor): input tensor containing the events (only one time bin)
        height (int): height of the image 
        width (int): width of the image
        batch_nu (int): specifies which sequence within the batch shall be plotted
        title (str): title of plot window
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        waitKey (int): waitKey for cv2 plot [0,1]
        dt (float): size of time bins (seconds)
    '''

    #Saving data to cpu 
    Input = Input.cpu()
    
    #Creating image 
    blank_image = np.zeros((height, width, 3), np.uint8)
    
    #Colouring on and off events 
    blank_image[Input[batch_nu, 1].to_sparse().coalesce().indices()[0], Input[batch_nu, 1].to_sparse().coalesce().indices()[1]] = (0,255,0)     # (B, G, R)
    blank_image[Input[batch_nu, 0].to_sparse().coalesce().indices()[0], Input[batch_nu, 0].to_sparse().coalesce().indices()[1]] = (0,0, 255)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) 
    cv2.moveWindow(title, *position) 
    cv2.imshow(title, blank_image)
    cv2.waitKey(waitKey)


def plot_histogram(
    input: torch.tensor, 
    nu_bins: int, 
    min: float, 
    max: float, 
    start: int, 
    xlabel: str,
    device: torch.device, 
    title = "histogram"
):
    '''This function plots the distribution of the weights in the input weight tensor in a histogram. 
    
    Parameters: 
        input (torch.Tensor): input for which distrubution shall be plotted
        nu_bins (int): number of bins used n the histogram
        min (int): lower end of range for which histogram is made
        max(int): higher end of range for which histogram is made
        start (float): first bin that is displayed in histogram
        xlabel (str): description of xlabel
        device (torch.device) device on which computations shall be performed
        title (str): title of the figure
    '''
    
    #Computing histogram
    histWd = torch.histc(input, nu_bins, min, max)

    #Finding maximum weights 
    max_Wd = torch.max(input)
    
    #Setting up array with weight values
    Wd_values = torch.arange(nu_bins).to(device) * max_Wd/nu_bins
    
    #Plotting histogram
    plt.figure()
    plt.plot(Wd_values[start:].cpu(), histWd[start:].cpu())
    plt.ylabel("Number of occurences [-]")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid()
    #plt.pause(0)

def normalize(
    X: torch.tensor,
    dim = None
):  
    '''This function normalizes the values of the input tensor and returns a tensor with the values scaled between [0,1]. 
    
    Parameters: 
       X (torch.Tensor): tensor whose values shall be normalized 
        dim (tuple): dimensions along which to normalize
    '''
    if dim == None:
        dim = tuple(np.arange(X.ndim))

    Xmax = torch.amax(X, dim = dim, keepdim=True)
    Xmin = torch.amin(X, dim = dim, keepdim=True)
    X = (X - Xmin)/(Xmax - Xmin)

    # Input_min = torch.amin(Input, dim = (2,3) , keepdim=True)
    # Input_max = torch.amax(Input, dim = (2,3), keepdim=True)
    # Input_mean = (Input - Input_min)/(Input_max - Input_min)

    return X


def plot_weights_SSConv(
    weights: torch.tensor, 
    title: str, 
    position: tuple, 
    size: tuple, 
    normalized: bool, 
    waitKey: int,
    device: torch.device
):
    '''This function plots the weights of the SSConv layer. 
    
    Parameters: 
        weights (torch.Tensor): weights 
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        normalized (bool): specifies whether or not to normalize weights
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''


    #Computing mean of weights to make them visible in plot

    if normalized:
        W_mean = normalize(weights, dim = (1,2,3,4))
    else:
        W_mean = weights
    
    #Unsqueezing to add colour channels 
    W_mean = W_mean.unsqueeze(-1).expand(-1,-1,-1,-1,-1, 3)
    
    #Colouring on events green 
    W_on = torch.tensor([0,255,0]).to(device)*W_mean[:,1]
    
    #Colouring off events red
    W_off = torch.tensor([0,0,255]).to(device)*W_mean[:,0]
    
    #Combining on and off events into one plot
    W_mean = W_on + W_off

    #Applying padding around filters to create visual separation 
    weights = torch.nn.functional.pad(W_mean, (0,0,1, 1, 1, 1), value = 255)
    
    #Flatten weights into 2-dimensional image for plotting 
    weights = np.array(weights.squeeze().reshape(2, int(weights.shape[0]/2), *weights.shape[2:]).permute(0,2,1,3,4).reshape(weights.shape[2]*2,weights.shape[3]*int(weights.shape[0]/2), 3).cpu()).astype('uint8')
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #1050 , 190
    cv2.moveWindow(title, *position) #1, 462
    cv2.imshow(title, weights)
    cv2.waitKey(waitKey)


    return weights


def plot_weights_MSConv(
    weights: torch.tensor, 
    nu_x: int, 
    nu_y: int, 
    title: str, 
    position: tuple, 
    size: tuple, 
    normalized: bool, 
    waitKey: int,
    device: torch.device
):
    '''This function plots the weights of the SSConv layer. 
    
    Parameters: 
        weights (torch.Tensor): weights 
        nu_x (int): number of windows depicting maps in x direction 
        nu_y (int): number of windows depicting maps in y direction
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        normalized (bool): specifies whether or not to normalize weights
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''


    #Computing mean of weights to make them visible in plot

    if normalized:
        W_mean = normalize(weights)
    else:
        W_mean = weights

    #Applying padding around filtersto create visual separation 
    weights = torch.nn.functional.pad(W_mean, (1, 1, 1, 1), value = 1)
    
    #Flatten weights into 2-dimensional image for plotting 
    weights = weights.squeeze().reshape(nu_x, nu_y, *weights.shape[2:])
    weights = np.array(weights.permute(0,2,1,3).reshape(weights.shape[0]*weights.shape[2],weights.shape[1]*weights.shape[3]).cpu())

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #1050 , 190
    cv2.moveWindow(title, *position) #1, 462
    cv2.imshow(title, weights)
    cv2.waitKey(waitKey)

def plot_weights_MSConv_sorted(    
    weights: torch.tensor, 
    title: str, 
    position: tuple, 
    size: tuple, 
    normalized: bool, 
    waitKey: int,
    device: torch.device
):
    '''This function plots the weights of the MSConv layer in the order of increasing OF magnitude. 
    
    Parameters: 
        weights (torch.Tensor): weights sorted by increasing OF magnitude
        nu_x (int): number of windows depicting maps in x direction 
        nu_y (int): number of windows depicting maps in y direction
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        normalized (bool): specifies whether or not to normalize weights
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''

     #Computing mean of weights to make them visible in plot

    if normalized:
        W_mean = normalize(weights)
    else:
        W_mean = weights

    #Applying padding around filters to create visual separation 
    weights = torch.nn.functional.pad(W_mean, (1, 1, 1, 1), value = 1)
    
    #Flatten weights into 2-dimensional image for plotting, delays in rows, maps in columns
    weights = np.array(weights.squeeze(1).permute(1,2,0,3).reshape(weights.shape[2]*weights.shape[3],weights.shape[0]*weights.shape[4]).cpu())

    # #Flatten weights into 2-dimensional image for plotting maps in rows, delays in columns 
    # weights = np.array(weights.squeeze(1).permute(0,2,1,3).reshape(weights.shape[0]*weights.shape[3],weights.shape[2]*weights.shape[4]).cpu())

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #1050 , 190
    cv2.moveWindow(title, *position) #1, 462
    cv2.imshow(title, weights)
    cv2.waitKey(waitKey)

def coloured_spikes(
    z: torch.tensor, 
    map_colours: list, 
    batch_nu: int, 
    delay_nu: int,
    device: torch.device
):
    '''This function takes the output spikes of a SNN layer as an input and multplies each map with a tuple corresponding to a different colour. 
    
    Parameters: 
        z (torch.Tensor): output spikes 
        map_colours (list): list of tuples specifying the colour of spikes in each map
        batch_nu (int): specifies which sequence within batch shall be plotted
        delay_nu (int): specifies which of the multisynaptic synapses shall be plotted 
        device (torch.device) device on which computations shall be performed
    '''

  
    
    #Transform colors of maps to format that can be multiplied with z 
    map_colours = (torch.tensor(map_colours)).unsqueeze(1).unsqueeze(2).expand(-1, z.shape[-2], z.shape[-1], -1).to(device)

    test0 = z[batch_nu, :, delay_nu].unsqueeze(-1).expand(-1,-1,-1,3)
    
    #Multiplying map colours with z to make sure that the spikes of each layer have a different colour 
    z = map_colours[:z.shape[1]]*(z[batch_nu, :, delay_nu]).unsqueeze(-1).expand(-1,-1,-1,3)

    return z


def plot_output_spikes_separate(
    z: torch.tensor, 
    map_colours: list, 
    batch_nu: int, 
    delay_nu:int,
    nu_x: int, 
    nu_y: int, 
    title: str, 
    position: tuple, 
    size: tuple,
    waitKey: int,
    device: torch.device
):
    '''This function plots the output spikes of a neuron layer with spikes from different maps depicted in different windows. 
    
    Parameters: 
        z (torch.Tensor): output spikes 
        map_colours (list): list of tuples specifying the colour of spikes in each map
        batch_nu (int): specifies which sequence within batch shall be plotted
        delay_nu (int): specifies which of the multisynaptic synapses shall be plotted 
        nu_x (int): number of windows depicting maps in x direction 
        nu_y (int): number of windows depicting maps in y direction
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''

    #Make colour legend
    z = torch.ones_like(z)
    
    #Adding different colour to each map 
    spikes = coloured_spikes(z, map_colours, batch_nu, delay_nu, device)
   
    #Applying padding for visual borders between maps
    spikes = torch.nn.functional.pad(spikes, (0,0, 1, 1, 1, 1), value = 0)

    #Flatten spike tensor to one 2D image
    spikes  = spikes.reshape(nu_x, nu_y, *spikes.shape[1:]).permute(0,2,1,3, 4).reshape(nu_x*spikes.shape[1], nu_y*spikes.shape[2], 3)
    spikes = np.array(spikes.cpu()).astype('uint8')

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #525,250
    cv2.moveWindow(title, *position) #527, 685
    cv2.imshow(title, spikes)
    cv2.waitKey(waitKey)


def plot_SSConv_colour_legend(
    weights: torch.tensor, 
    title: str, 
    position: tuple, 
    size: tuple, 
    normalized: bool, 
    waitKey: int,
    device: torch.device,
    colour_map = None
):
    '''This function provides a legend for the colours of the SSConv output. 
    
    Parameters: 
        weights (torch.Tensor): weights 
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        normalized (bool): specifies whether or not to normalize weights
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
        colour_map (torch.tensor): colour map of the neuron layer
    '''


    #Computing mean of weights to make them visible in plot

    if normalized:
        W_mean = normalize(weights)
    else:
        W_mean = weights
    
    #Unsqueezing to add colour channels 
    W_mean = W_mean.unsqueeze(-1).expand(-1,-1,-1,-1,-1, 3)
    
    #Colouring on events green 
    W_on = torch.tensor([0,255,0]).to(device)*W_mean[:,1]
    
    #Colouring off events red
    W_off = torch.tensor([0,0,255]).to(device)*W_mean[:,0]
    
    #Combining on and off events into one plot
    W_mean = W_on + W_off

    #Add extra dimension for legend 
    W_new = W_mean.expand(-1,2,-1,-1,-1).clone().detach()

    W_new[:, 0, :, :] = torch.tensor(colour_map).unsqueeze(1).unsqueeze(2).expand(-1, W_mean.shape[2], W_mean.shape[3], -1)

    #Applying padding around filters to create visual separation 
    weights = torch.nn.functional.pad(W_new, (0,0,1, 1, 1, 1), value = 255)
    
    #Flatten weights into 2-dimensional image for plotting 
    weights = np.array(weights.permute(1,2,0,3,4).reshape(weights.shape[2]*weights.shape[1], weights.shape[3]*weights.shape[0], 3).cpu()).astype('uint8')
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #1050 , 190
    cv2.moveWindow(title, *position) #1, 462
    cv2.imshow(title, weights)
    cv2.waitKey(waitKey)



def plot_output_spikes_together(
    z: torch.tensor, 
    map_colours: list, 
    batch_nu: int,  
    delay_nu: int,
    title: str, 
    position: tuple, 
    size: tuple,
    waitKey: int,
    device: torch.device
):
    '''This function plots the output spikes of a neuron layer with spikes from different maps depicted in different windows. 
    
    Parameters: 
        z (torch.Tensor): output spikes 
        map_colours (list): list of tuples specifying the colour of spikes in each map
        batch_nu (int): specifies which sequence within batch shall be plotted
        delay_nu (int): specifies which of the multisynaptic synapses shall be plotted 
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''

   

    #Adding different colour to each map 
    spikes = coloured_spikes(z, map_colours, batch_nu, delay_nu, device)

    #Converting to numpy array for plotting
    spikes = np.array(spikes.cpu()).astype('uint8')
    
    #Plotting   
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size)
    cv2.moveWindow(title, *position) 
    add_spikes = np.zeros_like(spikes[0])
    for map in range(len(spikes)):
        #Determine where spikes occur in current map
        mask = np.where(spikes[map] != np.array([0,0,0]))[0:2]
        #Putting image of spikes of current map on top of overall image
        add_spikes[mask[0], mask[1]] = spikes[map, mask[0], mask[1]]
    cv2.imshow(title, add_spikes)
    cv2.waitKey(waitKey)

    return add_spikes

def plot_output_spikes_cube(
    spike_indices: torch.tensor,
    height: int,
    width: int, 
    maps: int, 
    s: torch.tensor, 
    k: torch.tensor,
    colour_map: list,
    title: str
):
    
    '''This function provides a 3D-image of the output spikes. 
    
    Parameters: 
        spike_indices (torch.tensor): indices of output spikes 
        width (int): width of the image in layer
        height (int): height of the image in layer 
        maps (int): number of maps in layer
        s(torch.tensor): stride of neuron layer
        k (torch.tensor): kernel size of neuron layer 
        colour_map (list): list containing colours for each map
        title (str): title of the figure
    '''
    #set up figure
    fig = plt.figure(num = title, figsize=(25,25))
    plt.clf()
    ax = fig.gca(projection ='3d')
    ax.set_xlabel("x-pixel [-]")
    ax.set_ylabel("Map number [-]")
    ax.set_zlabel("y-pixel [-]")
    ax.grid(False)
    ax.invert_zaxis()
    
    #Stretch y-axis so maps can be seen more easily 
    x_scale = 1
    y_scale = 1.5
    z_scale = 1

    scale =np.diag([x_scale, y_scale, z_scale, 1.0])
    # scale=scale*(1.0/scale.max())
    # scale[3,3]=1.0

    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj
    
    #Define corners of neuron layer
    r_width = [0, width]
    r_maps = np.arange(maps)
    r_height = [0, height]

    #Loop through all connections of the corner points
    for i, j in combinations(np.array(list(product(r_width, r_maps, r_height))), 2):
        if np.sum(np.abs(i-j).astype(bool)) == 1:
            ax.plot3D(*zip(i, j), color="gray", linewidth = 1)

    #extract spike coordinates
    spike_x = np.array(spike_indices[3, :].cpu())              #width
    spike_y = np.array(spike_indices[1, :].cpu())              #map
    #spike_z = np.array(height - 1 - spike_indices[2, :].cpu()) #height
    spike_z = np.array(spike_indices[2, :].cpu()) #height

    #extract colours for each spike corresponding to spiking map
    colours  = (np.array(colour_map)[spike_y.astype(int)]/255)[:, ::-1]

    #plot spikes
    ax.scatter(spike_x, spike_y, spike_z, c = colours, linewidths = 3)

    # #plot spike windows 
    # window_x = np.array(torch.stack(([boxl[:,0], boxu[:,0]]), dim = 1).cpu()).astype(int)
    # window_y = [0, maps]
    # window_z = height -1 - np.array(torch.stack(([boxl[:,1], boxu[:,1]]), dim = 1).cpu()).astype(int)

    #plot window defining neurons with shared PST 
    s = np.array(s.cpu())
    k = np.array(k.cpu())

    window_x = np.array([spike_x - (k[2]-1)/s[2], spike_x + (k[2]-1)/s[2]])
    window_y = [0, maps]
    window_z = np.array([spike_z - (k[1]-1)/s[1], spike_z + (k[1]-1)/s[1]])

    for i in range(len(spike_x)):
        for i, j in combinations(np.array(list(product(window_x[:,i], window_y, window_z[:,i]))), 2):
            if np.sum(np.abs(i-j).astype(bool)) == 1:
                ax.plot3D(*zip(i, j), color="green")
    
    ax.set_title(title)
    

    #plt.pause(1)
   
    #plt.show()

def plot_voltage_trace(
    v_decayed: torch.tensor, 
    batch_num: int,
    map_num: int,
    title: str, 
    position: tuple, 
    size: tuple, 
    waitKey: int, 
    device: torch.device
):

    '''This function plots the voltage trace of the SSConv layer. 
    
    Parameters: 
        v_decayed (torch.Tensor): decayed voltage for each neuron of the SSConv layer 
        batch_num (int): batch for which voltage trace shall be plotted
        map_num (int): map number for which voltage trace shall be plotted
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''

    #Normalize  voltage trace 
    voltage_trace = normalize(v_decayed)

    voltage_trace = 255*voltage_trace[batch_num, map_num].unsqueeze(-1).expand(-1,-1, 3)
    
    #Converting to numpy array for plotting
    voltage_trace = np.array(voltage_trace.cpu()).astype('uint8')
    
    #Plotting   
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size)
    cv2.moveWindow(title, *position) 
    cv2.imshow(title, voltage_trace)
    cv2.waitKey(waitKey)






def plot_presynaptic_traces(
    X: torch.tensor, 
    map_colours: list,
    nu_x: int, 
    nu_y: int,
    boxes: torch.tensor,
    box_indices: torch.tensor,
    add_windows: bool, 
    batch_nr: int, 
    title: str, 
    position: tuple, 
    size: tuple,
    waitKey: int,
    device: torch.device
):

    '''This function plots the output spikes of a neuron layer with spikes from different maps depicted in different windows. 
    
    Parameters: 
        X(torch.Tensor): presynaptic trace
        map_colours (list): list of tuples specifying the colour of spikes in each map
        nu_x (int): number of windows depicting maps in x direction 
        nu_y (int): number of windows depicting maps in y direction
        boxes (torch.tensor): presynaptic windows of spikes
        box_indices (torch.tensor):
        add_windows(bool): specifies whether or not to also plot windows
        batch_nr (int): specifies which sequence within the batch shall be plotted
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''
   
    # X[0,1,0,1,8] = 0
    pst = (X[batch_nr] - torch.min(X[batch_nr]))/(torch.max(X[batch_nr]) - torch.min(X[batch_nr]))
    # pst = X[batch_nr]
    pst = pst.squeeze()
    pst = pst.unsqueeze(-1).expand(-1, -1,-1, 3)
    pst = pst.clone()
    pst = torch.nn.functional.pad(255*(pst), (0,0,1, 1, 1, 1), value = 255)
    pst = pst.reshape(nu_x,nu_y,*pst.shape[1:])
    pst = np.array(pst.permute(0,2,1,3,4).reshape(pst.shape[0]*pst.shape[2],pst.shape[1]*pst.shape[3],3).cpu()).astype('uint8')
    

    #Adding presynaptic trace windows
    #TODO: Make suitable for MSConv
    if add_windows:
        for i in range(len(boxes)):
            map = box_indices[1, i]
            # test2 = tuple(boxes[i,2:] + np.array([4*len(pst[0])/5 + 1, len(pst)/2 + 1]).astype(int))
            # test1 = tuple(boxes[i, :2] + np.array([4*len(pst[0])/5, len(pst)/2]).astype(int))
            # pst = cv2.rectangle(pst, tuple(boxes[i, :2]-1 + 1), tuple(boxes[i,2:] +1 ), (0,255,0), 1)
            # pst = cv2.rectangle(pst, test1, test2, (0,255,0), 1)
            test2 = tuple(boxes[i,2:] + np.array([len(pst[0])/2 + 1, 1]).astype(int))
            test1 = tuple(boxes[i, :2] + np.array([len(pst[0])/2, 0]).astype(int))
            #Plot boxes around pst in on events. One pixel is added since padding was applied to PST. One pixel is subtracted and added to lowet and upper limits, respectively
            pst = cv2.rectangle(pst, tuple(boxes[i, :2]-1 + 1), tuple(boxes[i,2:] +1), map_colours[map], 1)
            pst = cv2.rectangle(pst, test1, test2,  map_colours[map], 1)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #525,250
    cv2.moveWindow(title, *position) #1, 685
    cv2.imshow(title, pst)
    cv2.waitKey(waitKey)

    return pst




def rotate_buffer(
    z: torch.tensor, 
    buffer: torch.tensor
):
    '''This function adds the latest entry (z) to a buffer (buffer) and removed the oldest one
    
    Parameters: 
        z (torch.tensor): new entry 
        map_colours (torch.tensor): buffer
    '''
    #Writing into buffer 
    buffer = torch.cat((z.unsqueeze(0), buffer[:-1]), dim = 0)   

    #Taking sum of buffer
    buffer_sum = torch.clip(torch.sum(buffer, dim = 0), 0, 1)  

    return buffer, buffer_sum        


def plot_pst_windwos(
    X: torch.tensor, 
    spike_indices: torch.tensor,
    title: str, 
    position: tuple, 
    size: tuple,  
    waitKey: int,
    device: torch.device
):
    '''This function creates a plot showing the presynaptic trace windows for every spike in each map. 
    
    Parameters: 
        X (torch.Tensor): presynaptic traces
        spike_indices (torch.tensor): tensor containing coordinates of spikes
        title (str): title of the plot
        position (tuple): position of plot (x and y coordinates)
        size (tuple): size of plot (x and y length in pixels)
        waitKey (int): waitKey for cv2 plot [0,1]
        device (torch.device) device on which computations shall be performed
    '''

    
    #Compute number of spikes in each layer 
    layers, num_sp_l = torch.unique(spike_indices[1], return_counts = True)
 
    #Applying padding to pst windows with singular dimensions removed (singular dimensions are different for SSConv and MSConv, that's why dimensions 1 and 2 are squeezed)
    pst = torch.nn.functional.pad(X.squeeze(1).squeeze(2), (1, 1, 1, 1), value = 1)

    
    #Plotting spikes of different maps in different rows 
    #Splitting the pst windows into groups corresponding to the map in which they occured
    pst = torch.split(pst, tuple(num_sp_l))

    #Applying padding to make the l
    pst = torch.nn.utils.rnn.pad_sequence(pst, batch_first=True, padding_value=1)
    pst = pst.permute(0, 2, 3, 1, 4)
    pst = pst.reshape(pst.shape[0]*pst.shape[1]*pst.shape[2], pst.shape[3]*pst.shape[4])
    pst = np.array(pst.cpu())

    # #Plotting all spikes of different maps in one rwo 
    # pst = pst.permute(1,2,0,3).reshape(pst.shape[1]*pst.shape[3], pst.shape[0]*pst.shape[2])
    # pst = np.array(pst.cpu())
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, size) #1050 , 190
    cv2.moveWindow(title, *position) #1, 462
    cv2.imshow(title, pst)
    cv2.waitKey(waitKey)










#Input data 
#Weights 
#Outputs 
#Presynaptic traces 

    