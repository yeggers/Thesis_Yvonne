import torch 


class SNN_param_noSSConv:
    """ In this class the SNN network parameters for the different kinds of input data are defined.

    Parameters: 
        data (string): data which is used for training. Use data = 'lines' for line data, data = 'checkerboard' to train/ infere with 
                       checkerboard data, data = 'rot_disk' for rotating disk data and data = 'roadmap' 
                       for roadmap data.
     
    """

    def __init__(self, data):
        self.data = data
        self.data_lines = {
                            'directory': 'data_tensors/lines',
                            'directory_gt': 'data_tensors/ground_truths/lines',
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(3.0), 
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                },
                            'Merge':{
                                'r': 1, 
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005,  
                                'alpha': 0
                                }, 
                            'MSConv':{
                                'r': 7, 
                                's': 2, 
                                'p': 0,
                                'w_init_ex' : 0.5, 
                                'w_init_inh' : -0.5, 
                                'f': 16, 
                                'm': 10, 
                                'tau_min': 1, 
                                'tau_max': 17,
                                'v_th': 1.3, 
                                'lambda_i': 0.003, 
                                'lambda_X': 0.003,
                                'alpha': 0.337, 
                                'beta': 0.5
                                }, 
                            'Pooling':{
                                'r': 8, 
                                's': 8, 
                                'p': 0,
                                'f': 16, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                },
                            'Dense':{
                                'f': 16 , 
                                'm': 1, 
                                'w_init':0.5,
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.5, 
                                'lambda': 0.005, 
                                'alpha': 0.25
                                }, 
                            'training':{ 
                                'n': torch.as_tensor(10**(-4)),  
                                'a' : torch.as_tensor(0),          
                                'L': torch.as_tensor(5*10**(-2)),   
                                'ma_len': torch.as_tensor(1)
                                }
                             }
        self.data_checkerboard = {
                            'directory': 'data_tensors/checkerboard',
                            'directory_gt': 'data_tensors/ground_truths/checkerboard',
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(3.0), 
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                },
                            'Merge':{
                                'r': 1, 
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005,  
                                'alpha': 0
                                }, 
                            'MSConv':{
                                'r': 7, 
                                's': 2, 
                                'p': 0,
                                'w_init_ex' : 0.5, 
                                'w_init_inh' : -0.5, 
                                'f': 16, 
                                'm': 10, 
                                'tau_min': 1, 
                                'tau_max': 200,
                                'v_th': 0.5, 
                                'lambda_i': 0.007, 
                                'lambda_X': 0.007,
                                'alpha': 0.224, 
                                'beta': 0.5
                                }, 
                            'Pooling':{
                                'r': 8, 
                                's': 8, 
                                'p': 0,
                                'f': 16, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                },
                            'Dense':{
                                'f': 16 , 
                                'm': 1, 
                                'w_init':0.5,
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.5, 
                                'lambda': 0.005, 
                                'alpha': 0.25
                                }, 
                            'training':{ 
                                'n': torch.as_tensor(10**(-4)),  
                                'a' : torch.as_tensor(0),          
                                'L': torch.as_tensor(5*10**(-2)),   
                                'ma_len': torch.as_tensor(1)
                                }
                             }
        self.data_rot_disk = {
                            'directory': 'data_tensors/rotating_disk/use',
                            'directory_gt': 'data_tensors/ground_truths/rot_disk',
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(1.0),
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                },
                            'Merge':{
                                'r': 1, 
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.0000, 
                                'lambda': 0.0025, 
                                'alpha': 0
                                }, 
                            'MSConv':{
                                'r': 5, 
                                's': 2, 
                                'p': 0,
                                'w_init_ex' : 0.5, 
                                'w_init_inh' : -0.5, 
                                'f': 64, 
                                'm': 10, 
                                'tau_min': 1, 
                                'tau_max': 400,
                                'v_th': 1, 
                                'lambda_i': 0.009, 
                                'lambda_X': 0.09,
                                'alpha': 0.05,
                                'beta': 0.5
                                }, 
                             'Pooling':{
                                 'r': 6, 
                                's': 6, 
                                'p': 0,
                                'f': 64, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                },
                            'Dense':{
                                'f': 16 , 
                                'm': 1, 
                                'w_init': 0.5,
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.3, 
                                'lambda': 0.03, 
                                'alpha': 0.1
                                }, 
                            'training':{ 
                                'n': torch.as_tensor(10**(-4)),  
                                'a' : torch.as_tensor(0),          
                                'L': torch.as_tensor(5*10**(-2)),   
                                'ma_len': torch.as_tensor(1)
                                }
                             }
        
        self.data_roadmap = {
                            'directory': 'data_tensors/roadmaps',
                            'directory_gt': 'data_tensors/ground_truths/roadmap',
                            'height': 264, 
                            'width': 320,
                            'delta_ref': torch.as_tensor(1.0) , 
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                }, 
                            'Merge':{
                                'r': 1,
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                'alpha': 0
                                }, 
                            'MSConv':{
                                'r': 5, 
                                's': 2, 
                                'p': 0,
                                'w_init_ex' : 0.5, 
                                'w_init_inh' : -0.5, 
                                'f': 64, 
                                'm': 10, 
                                'tau_min': 1, 
                                'tau_max': 25,
                                'v_th': 1.5, 
                                'lambda_i': 0.002, 
                                'lambda_X': 0.002,
                                'alpha': 0.25, 
                                'beta': 0.5
                                }, 
                            'Pooling':{
                                'r': 8, 
                                's': 8, 
                                'p': 0,
                                'f': 64, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                },
                            'Dense':{
                                'f': 32, 
                                'm': 1, 
                                'w_init': 0.5,
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.4, 
                                'lambda': 0.015, 
                                'alpha': 0.25
                                }, 
                            'training':{
                                'n': torch.as_tensor(10**(-4)),  
                                'a' : torch.as_tensor(0),          
                                'L': torch.as_tensor(5*10**(-2)),   
                                'ma_len': torch.as_tensor(1)
                                }
                             }
        self.data_ODA_dataset = {
                            'directory': 'data_tensors/ODA_dataset',
                            'directory_gt': 'data_tensors/ground_truths/ODA_dataset',
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(1.0) , 
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                }, 
                            'SSConv':{
                                'r': 5, 
                                's': 2, 
                                'p': 0,
                                'w_init': 0.5,
                                'f': 16, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.4, 
                                'lambda': 0.005, 
                                'alpha': 0.25
                                }, 
                            'Merge':{
                                'r': 1,
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                }, 
                            'MSConv':{
                                'r': 5, 
                                's': 2, 
                                'p': 0,
                                'w_init_ex' : 0.5, 
                                'w_init_inh' : -0.5, 
                                'f': 64, 
                                'm': 10, 
                                'tau_min': 1, 
                                'tau_max': 25,
                                'v_th': 0.4, 
                                'lambda_i': 0.015, 
                                'lambda_X': 0.015,
                                'alpha': 0.25, 
                                'beta': 0.224
                                }, 
                            'Pooling':{
                                'r': 8, 
                                's': 8, 
                                'p': 0,
                                'f': 64, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda': 0.005, 
                                },
                            'Dense':{
                                'f': 32, 
                                'm': 1, 
                                'w_init': 0.5,
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.4, 
                                'lambda': 0.015, 
                                'alpha': 0.25
                                }, 
                            'training':{
                                'n': torch.as_tensor(10**(-4)),  
                                'a' : torch.as_tensor(0),          
                                'L': torch.as_tensor(5*10**(-2)),   
                                'ma_len': torch.as_tensor(1)
                                }
                             }
        
        
       

    class define_SNN_param:
        """ In this class the SNN parameters are defined based on the provided input parameters.

        Parameters: 
            parameters (dict): dictionary specifying the parameters. The dictionary must contain the following entries:
                                    - directory (str): directory in which training data can be found
                                    - height (int): height of downsampled input image [pixels]
                                    - width (int): width of downsampled input image [pixels] 
                                    - delta_ref (torch.tensor):  Refractory period for all neurons [ms]
                                    
                                    - Input.r (int): kernel size of Input layer [-]
                                    - Input.s (int): stride for Input layer [-]
                                    - Input.p (int): padding for Input layer [-]
                                    - Input.w_init (float): initial value of weights for training in Input layer[-]
                                    - Input.f (int): number of maps in Input layer [-]
                                    - Input.m (int): number of synapses between any two neurons in Input layer [-]
                                    - Input.in (int): number of channels fed into input layer [-]
                                    
                                    - Merge.r (int): kenerl size of Merge layer [-]
                                    - Merge.s (int): stride for Merge layer [-]
                                    - Merge.p (int): padding for Merge layer [-]
                                    - Merge.f (int): number of maps in Merge layer [-]
                                    - Merge.m (int): number of synapses between any two neurons in Merge layer [-]
                                    - Merge.tau (torch.tensor long): synaptic delay of neurons in Merge layer [ms]
                                    - Mergev.v_th (float): voltage threshold for spiking of neurons in Merge layer [-]
                                    - Merge.lambda (float): synaptic, membrane and presynaptic trace constant in Merge layer [s]
                                    
                                    - MSConv.r (int): kenerl size of MSConv layer [-]
                                    - MSConv.s (int): stride for MSConv layer [-]
                                    - MSConv.p (int): padding for MSConv layer [-]
                                    - MSConv.w_init_ex (float): initial value of excitatory weights for training in MSConv layer[-]
                                    - MSConv.w_init_inh (float): initial value of inhibitory weights for training in MSConv layer[-]
                                    - MSConv.f (int): number of maps in MSConv layer [-]
                                    - MSConv.tau_min (int): minimum delay in multisynaptic connections in MSConv layer [ms]
                                    - MSConv.tau_max (int): maximum delay in multisynaptic connections in MSConv layer [ms]
                                    - MSConv.beta (float): weight of inhibitory weights [-]
                                    - MSConv.m (int): number of synapses between any two neurons in MSConv layer [-]
                                    - MSConv.tau (torch.tensor long): synaptic delay of neurons in MSConv layer [ms]
                                    - MSConvv.v_th (float): voltage threshold for spiking of neurons in MSConv layer [-]
                                    - MSConv.lambda (float): synaptic, membrane and presynaptic trace constant in MSConv layer [s]
                                    - MSConv.alpha (float): presynaptic trace decay factor in MSConv layer[-]
                                    
                                    - Pooling.r (int): kenerl size of Pooling layer [-]
                                    - Pooling.s (int): stride for Pooling layer [-]
                                    - Pooling.p (int): padding for Pooling layer [-]
                                    - Pooling.f (int): number of maps in Pooling layer [-]
                                    - Pooling.m (int): number of synapses between any two neurons in Pooling layer [-]
                                    - Pooling.tau (torch.tensor long): synaptic delay of neurons in Pooling layer [ms]
                                    - Poolingv.v_th (float): voltage threshold for spiking of neurons in Pooling layer [-]
                                    - Pooling.lambda (float): synaptic, membrane and presynaptic trace constant in Pooling layer [s]
                                    
                                    - Dense.f (int): number of maps in Dense layer [-]
                                    - Dense.m (int): number of synapses between any two neurons in Dense layer [-]
                                    - Dense.w_init(float): initial value of weights for training in Dense layer[-]
                                    - Dense.tau (torch.tensor long): synaptic delay of neurons in Dense layer [ms]
                                    - Densev.v_th (float): voltage threshold for spiking of neurons in Dense layer [-]
                                    - Dense.lambda (float): synaptic, membrane and presynaptic trace constant in Dense layer [s]
                                    - Dense.alpha (float): presynaptic trace decay factor in Dense layer[-]
                                    
                                    - training.n (torch.tensor): learning rate [-]
                                    - training.a (torch.tensor): factor to control spread of weights [-]
                                    - training.L (torch.tensor): value of stopping criterion [-]
                                    - training.ma_len(torch.tensor): length of moving average window to compute stopping criterion [-] 
                                    
                                 
    
        
        """

        def __init__(self, parameters):

            self.directory = parameters['directory']
            self.directory_gt = parameters['directory_gt']

            self.height = parameters['height']
            self.width = parameters['width']

            self.delta_ref = parameters['delta_ref']                                        #Refractory period for all neurons [ms]

            ##Input layer
            self.input_in_dim = parameters['Input']['in']                                   #Number of input channels 
            self.input_out_dim = parameters['Input']['f']                                   #Number of output channels
            self.input_m = parameters['Input']['m']                                         #Number of synapses between two neurons
            self.input_kernel = parameters['Input']['r']                                    #Kernel size 
            self.input_stride = parameters['Input']['s']                                    #Stride 
            self.input_padding = parameters['Input']['p']                                   #Padding  
            
            #Only relevant if multisynaptic connections with input 
            self.input_kernel3D = (self.input_m, self.input_kernel, self.input_kernel)      #Kernel for 3D convolution 
            self.input_stride3D = (self.input_m, self.input_stride, self.input_stride)      #Stride for 3D convolution 
            self.input_padding3D = (0, self.input_padding, self.input_padding)              #Padding for 3D convolution 
            self.input_weights = torch.ones((self.input_out_dim, self.input_in_dim, self.input_kernel, self.input_kernel)) #Weight tensor to downsample input

            #TODO: check if parameters related to presynaptic trace are neccessary 
            ##Merge layer 
            self.merge_in_dim = self.input_out_dim                                         #Number of input channels 
            self.merge_out_dim = parameters['Merge']['f']                                   #Number of output channels 
            self.merge_kernel =  parameters['Merge']['r']                                   #Kernel size 
            self.merge_stride =  parameters['Merge']['s']                                   #Stride 
            self.merge_padding = parameters['Merge']['p']                                   #Padding 
            self.merge_m = parameters['Merge']['m']                                         #Number of synapses between two neurons
            self.merge_lambda_i = parameters['Merge']['lambda']                             #Synaptic time constant (Used in Norse LIF neuron)
            self.merge_lambda_v = parameters['Merge']['lambda']    
            self.merge_lambda_X = parameters['Merge']['lambda']                           #Membrane time constant 
            self.merge_v_th = parameters['Merge']['v_th']                                   #Voltage threshold for spikes 
            self.merge_delay = parameters['Merge']['tau']      
            self.merge_alpha = parameters['Merge']['alpha']                                #Synaptic delay of neurons [ms]
            
            self.merge_kernel3D = (self.merge_m, self.merge_kernel, self.merge_kernel)      #Kernel for 3D convolution 
            self.merge_stride3D = (self.merge_m, self.merge_stride, self.merge_stride)      #Stride for 3D convolution 
            self.merge_padding3D = (0, self.merge_padding, self.merge_padding)              #Padding for 3D convolution    
            self.merge_weights = torch.ones((self.merge_out_dim, self.merge_in_dim, self.merge_m, self.merge_kernel, self.merge_kernel))      #Weights for compuation of weighted spikes 
            self.merge_weights_pst = torch.ones((self.merge_out_dim, self.merge_in_dim, self.merge_m, self.merge_kernel, self.merge_kernel))  #Weights for computation of presynaptic trace

            ##MSConv layer: modified LIF neurons 
            self.MSConv_in_dim = self.merge_out_dim                                         #Number of input channels 
            self.MSConv_out_dim = parameters['MSConv']['f']                                 #Number of output channels 
            self.MSConv_kernel =  parameters['MSConv']['r']                                 #Kernel size 
            self.MSConv_stride =  parameters['MSConv']['s']                                 #Stride 
            self.MSConv_padding = parameters['MSConv']['p']                                 #Padding 
            self.MSConv_w_init_ex = parameters['MSConv']['w_init_ex']                       #Initial weights for excitatory connections
            self.MSConv_w_init_inh = parameters['MSConv']['w_init_inh']                     #Initial weights for inhibitory connections
            self.MSConv_m = parameters['MSConv']['m']                                       #Number of synapses between two neurons
            self.MSConv_lambda_i = parameters['MSConv']['lambda_i']                           #Synaptic time constant (Used in Norse LIF neuron)
            self.MSConv_lambda_v = parameters['MSConv']['lambda_i']                           #Membrane time constant 
            self.MSConv_lambda_X = parameters['MSConv']['lambda_X']                           #Presynaptic trace time constant    
            self.MSConv_v_th = parameters['MSConv']['v_th']                                 #Voltage threshold for spikes 
            self.MSConv_alpha = parameters['MSConv']['alpha']                               #Presynaptic trace growth factor  
            self.MSConv_beta = parameters['MSConv']['beta']                                 #Weight of inhibitory weights 
            self.MsConv_tau_min = parameters['MSConv']['tau_min'] 
            self.MsConv_tau_max = parameters['MSConv']['tau_max'] 
            
            self.MSConv_delay = torch.linspace(self.MsConv_tau_min, self.MsConv_tau_max, self.MSConv_m).to(dtype=torch.long)  #Synaptic delay of neurons [ms]
            self.MSConv_kernel3D = (self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel)   #Kernel for 3D convolution
            self.MSConv_stride3D = (self.MSConv_m, self.MSConv_stride, self.MSConv_stride)   #Stride for 3D convolution 
            self.MSConv_padding3D = (0, self.MSConv_padding, self.MSConv_padding)            #Padding for 3D convolution   
            self.MSConv_weight_shape = (self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel)  
            self.MSConv_weights_ex = self.MSConv_w_init_ex*torch.ones((self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel))    #Initial excitatory weights for compuation of weighted spikes 
            self.MSConv_weights_inh = self.MSConv_w_init_inh*torch.ones((self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel))  #Initial inhibitory Weights for compuation of weighted spikes 
            self.MSConv_weights_pst = torch.ones((self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel))  #Weights for computation of presynaptic trace

            #TODO: check if parameters related to presynaptic trace are neccessary 
            ##Pooling layer
            self.pooling_in_dim = self.MSConv_out_dim                                       #Number of input channels 
            self.pooling_out_dim =  parameters['Pooling']['f']                              #Number of output channels 
            self.pooling_kernel =  parameters['Pooling']['r']                               #Kernel size 
            self.pooling_stride =  parameters['Pooling']['s']                               #Stride 
            self.pooling_padding = parameters['Pooling']['p']                               #Padding 
            self.pooling_m = parameters['Pooling']['m']                                     #Number of synapses between two neurons
            self.pooling_lambda_i = parameters['Pooling']['lambda']                         #Synaptic time constant (Used in Norse LIF neuron)
            self.pooling_lambda_v = parameters['Pooling']['lambda']                         #Membrane time constant 
            self.pooling_lambda_X = parameters['Pooling']['lambda']                         #Presynaptic trace time constant    
            self.pooling_v_th = parameters['Pooling']['v_th']                               #Voltage threshold for spikes 
            self.pooling_delay = parameters['Pooling']['tau']                               #Synaptic delay of neurons [ms]
            
            self.pooling_kernel3D = (self.pooling_m, self.pooling_kernel, self.pooling_kernel)   #Kernel for 3D convolution 
            self.pooling_stride3D = (self.pooling_m, self.pooling_stride, self.pooling_stride)   #Stride for 3D convolution 
            self.pooling_padding3D = (0, self.pooling_padding, self.pooling_padding)             #Padding for 3D convolution    
            self.pooling_weights = torch.ones((self.pooling_out_dim, self.pooling_in_dim, self.pooling_m, self.pooling_kernel, self.pooling_kernel))      #Weights for compuation of weighted spikes 
            self.pooling_weights_pst = torch.ones((self.pooling_out_dim, self.pooling_in_dim, self.pooling_m, self.pooling_kernel, self.pooling_kernel))  #Weights for computation of presynaptic trace 



            ##Dense layer 
            self.dense_in_dim = self.pooling_out_dim                                        #Number of input channels 
            self.dense_out_dim = parameters['Dense']['f']                                   #Number of output channels 
            self.dense_w_init = parameters['Dense']['w_init']                               #Initial value for weights 
            self.dense_m = parameters['Dense']['m']                                             #Number of synapses between two neurons
            self.dense_lambda_i = parameters['Dense']['lambda']                             #Synaptic time constant (Used in Norse LIF neuron)
            self.dense_lambda_v = parameters['Dense']['lambda']                             #Membrane time constant 
            self.dense_lambda_X = parameters['Dense']['lambda']                             #Presynaptic trace time constant    
            self.dense_v_th = parameters['Dense']['v_th']                                   #Voltage threshold for spikes 
            self.dense_alpha = parameters['Dense']['alpha']                                 #Presynaptic trace growth factor  
            self.dense_delay = parameters['Dense']['tau']                                   #Synaptic delay of neurons [ms]
            
            self.dense_stride3d = (1,1,1)                                                   #Random large values to simulate fully connected layer   
            self.dense_padding3d = (0,0,0)
            #dense_weights = SSConv_w_init*torch.ones((SSConv_out_dim, SSConv_in_dim, SSConv_kernel, SSConv_kernel)) #Inital weight tensor
            #dense_weights_pst = torch.ones((SSConv_out_dim, SSConv_in_dim, SSConv_kernel, SSConv_kernel))           #Weights for computation of presynaptic trace


            ##Training parameters 
            self.training_n = parameters['training']['n']                                   #Learning rate 
            self.training_a = parameters['training']['a']                                   #Factor to control spread of weights
            self.training_L= parameters['training']['L']                                    #Stopping criterion 
            self.training_ma_len = parameters['training']['ma_len']                         #Length of moving average window to compute stopping criterion


            #Plotting (colours from https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors)
            self.map_colours = [
                        (0, 0, 255),      #red
                        (0, 255, 0),      #lime
                        (255, 0, 0),      #blue
                        (0, 255, 255),    #yellow
                        (255, 255, 0),    #cyan        
                        (255, 0, 255),    #magenta
                        (170, 205, 102),  #medium aqua marine
                        (0, 0, 128),      #maroon
                        (0, 128, 128),    #olive
                        (0, 128, 0),      #green
                        (128, 0, 128),    #purple
                        (128, 128, 0),    #teal
                        (128, 0, 0),      #navy
                        (192, 192, 192),  #silver
                        (128, 128, 240),  #light coral
                        (0, 140, 255)     #dark orange
                        ]    

        
        
        
    def define_data(self):
        """ This function returns an instance of the define_SNN_param class specifying the SNN parameters corresponding to the chosen training data."""

        if self.data == 'lines':
            return self.define_SNN_param(self.data_lines)
        
        if self.data == 'checkerboard':
            return self.define_SNN_param(self.data_checkerboard)
        
        elif self.data == 'rot_disk':
            return self.define_SNN_param(self.data_rot_disk)
        
        elif self.data == 'roadmap':
            return self.define_SNN_param(self.data_roadmap)
        
        elif self.data == 'ODA_dataset':
            return self.define_SNN_param(self.data_ODA_dataset)


        else: 
            print('Invalid selection')
        



