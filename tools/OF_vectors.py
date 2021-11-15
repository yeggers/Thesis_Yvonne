from tools.plotting import plot_histogram
import torch 
import SNN_param as par
from SNN_param import SNN_param
from scipy.stats import linregress
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats
import cv2




class compute_OF():

    def __init__(
        self, par, MSConv_kernels_exc, MSConv_kernels_inh, device, gamma = 0
    ):
        #Loading MSConv weights 
        self.MSConv_kernels_exc = MSConv_kernels_exc
        self.MSConv_kernels_inh = MSConv_kernels_inh
        self.gamma = gamma
        self.par = par
        self.beta = self.par.MSConv_beta

        #Combining excitatory and inhibitory weights 
        self.MSConv_kernels = self.MSConv_kernels_exc + self.beta * self.MSConv_kernels_inh


    def compute_OF(self):

        #Setting up tensor to collect OF results 
        OF = np.zeros((self.MSConv_kernels.shape[0], 2))
        OF_norm, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v = (0, 0, 0, 0, 0, 0, 0)

        
        #Determine max for each map 
        MSConv_kernels_max = torch.amax(self.MSConv_kernels, dim = (3,4), keepdim=True)

        #Determeine min sum for each map 
        MSConv_kernels_min = torch.amin(self.MSConv_kernels, dim = (3,4) , keepdim=True)

        #Normalize MSConv_kernels (for automatic parameter tuning)

        MSConv_kernels_norm = (self.MSConv_kernels - MSConv_kernels_min)/(MSConv_kernels_max - MSConv_kernels_min)
        
        
        
        
        #Compute sum of weights in each group of synapses
        MSConv_kernels_sum = torch.sum(self.MSConv_kernels, dim = (3,4))

        #Determine max sum for each map 
        MSConv_kernels_sum_max, __ = torch.max(MSConv_kernels_sum, dim = 2, keepdim=True)

        #Determeine min sum for each map 
        MSConv_kernels_sum_min, __ = torch.min(MSConv_kernels_sum, dim = 2, keepdim=True)

      
        #Normalize sums 
        MSConv_kernel_sum_norm = (MSConv_kernels_sum - MSConv_kernels_sum_min)/(MSConv_kernels_sum_max - MSConv_kernels_sum_min)

        #Do not compute OF if first and last delays of filters are the same 
        #if torch.all(~torch.isnan(MSConv_kernel_sum_norm)) and torch.all(~torch.isnan(MSConv_kernels_norm)):

        #Finding which sum of weights is larger than scaled maximum sum 
        condition = torch.ge(MSConv_kernel_sum_norm, self.gamma)

        #Determine indices of tau min and tau max 
        indices = torch.nonzero(condition)

        #Computing number of synaptic groups larger than scaled max for each map
        __, num_larger = torch.unique(indices[:,0], return_counts = True)

        #Splitting indices into bits corresponding to the different maps
        split_indices = torch.split(indices,tuple(num_larger))
        tau_min = [element[0,2] for element in split_indices]
        tau_max = [element[-1,2] for element in split_indices] 
        
    
        #Extracting histograms
        tuple0 = tuple(torch.arange(self.MSConv_kernels.shape[0]))
        tuple1 = tuple(torch.arange(self.MSConv_kernels.shape[1]))

       

        hist_tau_min_hor = torch.sum(self.MSConv_kernels[tuple0,tuple1,tau_min], dim = 1)
        hist_tau_min_ver = torch.sum(self.MSConv_kernels[tuple0,tuple1,tau_min], dim = 2)

        hist_tau_max_hor = torch.sum(self.MSConv_kernels[tuple0,tuple1,tau_max], dim = 1)
        hist_tau_max_ver = torch.sum(self.MSConv_kernels[tuple0,tuple1,tau_max], dim = 2)

        # hist_tau_min_hor = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_min], dim = 1)
        # hist_tau_min_ver = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_min], dim = 2)

        # hist_tau_max_hor = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_max], dim = 1)
        # hist_tau_max_ver = torch.sum(MSConv_kernels_norm[tuple0,tuple1,tau_max], dim = 2)

        #Computing difference in histograms 
        hist_hor = (hist_tau_min_hor - hist_tau_max_hor).squeeze() 
        hist_ver = -(hist_tau_min_ver - hist_tau_max_ver).squeeze()#Applying minus to make OF upwards positive 

        #Estimating slope with least squares (norm(Ax-B) = 0)
        A_hor = torch.arange(hist_hor.shape[-1]).expand(hist_hor.shape[0], -1)
        A_ver = torch.arange(hist_ver.shape[-1]).expand(hist_ver.shape[0], -1)

        linregress2D = np.vectorize(linregress, signature='(n),(n)->(),(),(),(),()')
        
        theta_u = linregress2D(np.array(A_hor), np.array(hist_hor.cpu()))
        theta_v = linregress2D(np.array(A_ver), np.array(hist_ver.cpu()))
        
        
        tau_max = torch.tensor(tau_max)
        tau_min = torch.tensor(tau_min)
        
        #Computing OF 
        OF[:,0] = theta_u[0]/(tau_max - tau_min)
        OF[:,1] = theta_v[0]/(tau_max - tau_min)

        #Normalizing OF according to vector length 
        OF_lengths = (OF[:,0]**2 + OF[:,1]**2)**0.5
        OF_lengths_max = np.max(OF_lengths)
     
        scale = 1/OF_lengths_max
        OF_norm = scale*OF

        # #Plot map 59
        # self.plot_histograms(59, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v)


        #OF_norm = OF
        
        return OF_norm, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v, OF

    def assign_MSConv_colour(self,colour_map, OF):
        #Loading colour  map 
        colour_map = cv2.imread(colour_map)

        #Determine indices of OF on colour map
        OF_indices_x = (OF[:,0]*(colour_map.shape[1]/2-1) + colour_map.shape[1]/2).astype(int)
        OF_indices_y = (-OF[:,1]*(colour_map.shape[0]/2-1) + colour_map.shape[0]/2).astype(int)

        #Finding colours 
        MSConv_colours = colour_map[OF_indices_y, OF_indices_x] 

        return MSConv_colours


    def plot_histograms(self,map_nu, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v):
        #Plotting results 
        plt.figure()
        plt.subplot(1,2,1)
        plt.bar(A_hor[map_nu], np.array(hist_hor[map_nu].cpu()))
        plt.plot(theta_u[1][map_nu]+ theta_u[0][map_nu]*A_hor[map_nu], 'r')
        plt.grid()
        plt.title("Horizontal histogram")
        plt.xlabel("X-neuron")
        plt.ylabel("Sum of weights")
        plt.subplot(1,2,2)
        plt.bar(A_ver[map_nu], np.array(hist_ver.cpu()[map_nu]))
        plt.plot(theta_v[1][map_nu]+ theta_v[0][map_nu]*A_ver[map_nu], 'r')
        plt.grid()
        plt.title("Vertical histogram")
        plt.xlabel("Y-neuron")
        plt.ylabel("Sum of weights")
        plt.show()

    def plot_OF(self, colourmap, OF):
        fig, ax = plt.subplots()
        img = cv2.imread(colourmap)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent = (-1,1,-1,1))
        ax.scatter(OF[:,0], OF[:,1], marker = 'x', c ='gray')
        n = np.arange(len(OF))

        #Print map numbers next to crosses
        for i, txt in enumerate(n):
            ax.annotate(txt, (OF[i,0], OF[i,1]), color ='white')
        plt.show()





if __name__ == "__main__":

    #Checking if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda:1")  
        print("Running on the GPU")

    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    #Loading trained MSConv weights 
    MSConv_kernels_exc = torch.load("MSConvWeights_exc.pt")
    MSConv_kernels_inh= torch.load("MSConvWeights_inh.pt")
    colour_map = 'colour_code.png'
    par = SNN_param('rot_disk').define_data()

    #Initializing class 
    OF_class = compute_OF(par, MSConv_kernels_exc, MSConv_kernels_inh, device, gamma = 0)
    #Computing OF
    OF, A_hor, A_ver, hist_hor, hist_ver, theta_u, theta_v = OF_class.compute_OF()
    #Plotting OF field 
    OF_class.plot_OF(colour_map,OF)
    OF_class.plot_histograms(0, A_hor,A_ver, hist_hor, hist_ver, theta_u, theta_v)
    MSConv_colours = OF_class.assign_MSConv_colour(colour_map, OF)
