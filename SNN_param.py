import torch 


class SNN_param:
    """ In this class the SNN network parameters for the different kinds of input data are defined.

    Parameters: 
        data (string): data which is used for training\inference. Use data = 'data_lines' to train/ infere with line data,
                       data = 'checkerboard' for checkerboard data, data = 'rot_disk' for rotating disk data and data = 'roadmap' 
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
                            'par_name': 'lines',
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                },
                            'SSConv':{
                                'r': 7, 
                                's': 1, 
                                'p': 0,
                                'w_init': 0.5,
                                'f': 4, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.2,  
                                'lambda_v': 0.005,
                                'lambda_X': 0.005,
                                'alpha': 0.4, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(16, 43, 58),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001])
                                }, 
                            'Merge':{
                                'r': 1, 
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001, 
                                'lambda_v': 0.005, 
                                'lambda_X': 0.005
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
                                'tau_max': 8,
                                'v_th': 4.9,  
                                'lambda_v': 0.004, 
                                'lambda_X': 0.004,
                                'alpha': 0.425, 
                                'beta': 0.5, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(64, 20, 27),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001]), 
                                'tau_max_gain': torch.as_tensor(1.5)
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
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(3.0), 
                            'par_name': 'checkerboard',
                            'Input':{ 
                                'r': 2, 
                                's': 2, 
                                'p': 0,
                                'f': 2, 
                                'm': 1, 
                                'in': 2
                                },
                            'SSConv':{
                                'r': 7, 
                                's': 1, 
                                'p': 0,
                                'w_init': 0.5,
                                'f': 4, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.2,  
                                'lambda_v': 0.005,
                                'lambda_X': 0.005,
                                'alpha': 0.4, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(16, 43, 58),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001])
                                }, 
                            'Merge':{
                                'r': 1, 
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001,
                                'lambda_v': 0.005,
                                'lambda_X': 0.005, 
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
                                'tau_max': 50,
                                'v_th': 0.5, 
                                'lambda_v': 0.005, 
                                'lambda_X': 0.005,
                                'alpha': 0.25, 
                                'beta': 0.5, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(64, 20, 27),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001]), 
                                'tau_max_gain': torch.as_tensor(1.5)
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
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(1),
                            'par_name': 'rot_disk',
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
                                'p': 2,
                                'w_init': 0.5,
                                'f': 32 , 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.32, 
                                'lambda_X': 0.005,
                                'lambda_v': 0.005,
                                'alpha':0.15, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(32, 45, 60),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(2), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001])
                                }, 
                            'Merge':{
                                'r': 1, 
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.0000, 
                                'lambda_v': 0.005, 
                                'lambda_X': 0.005
                                }, 
                            'MSConv':{
                                'r': 5, 
                                's': 2, 
                                'p': 2,
                                'w_init_ex' : 0.5, 
                                'w_init_inh' : -0.5, 
                                'f': 64, 
                                'm': 10, 
                                'tau_min': 1, 
                                'tau_max': 250,
                                'v_th': 0.2,  
                                'lambda_v': 0.03,
                                'lambda_X': 0.03,
                                'alpha': 0.1,
                                'beta': 0.5, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(64, 23, 30),1,(0,0,0)], 
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001]), 
                                'tau_max_gain': torch.as_tensor(1.5)
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
                                'ma_len': torch.as_tensor(250)
                                }
                             }
        
        self.data_roadmap = {
                            'directory': 'data_tensors/roadmap',
                            'height': 264, 
                            'width': 320,
                            'delta_ref': torch.as_tensor(1.0) , 
                            'par_name': 'roadmap',
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
                                'v_th': 0.3,
                                'lambda_X': 0.005, 
                                'lambda_v': 0.005,
                                'alpha': 0.25, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(16, 64, 78),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001])
                                },                             
                            'Merge':{
                                'r': 1,
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([0]).long(), 
                                'v_th': 0.001, 
                                'lambda_v': 0.005, 
                                'lambda_X': 0.005
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
                                'v_th': 0.6, 
                                'lambda_v': 0.03,
                                'lambda_X': 0.03,
                                'alpha': 0.1, 
                                'beta': 0.5, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(64, 30, 37),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'kernel_vth': 31, 
                                'stride_vth': 1, 
                                'padding_vth': 15, 
                                'vth_buffer': 20,
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': [0.0, 0.01, 0.00001],
                                'tau_max_gain': torch.as_tensor(1.5),
                                'target_paramters': [12.327922591384661, 7.516300463119961e-05, 1.796295780751845]
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
                            'height': 180, 
                            'width': 240,
                            'delta_ref': torch.as_tensor(1.0) , 
                            'par_name': 'ODA_dataset',
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
                                'lambda_X': 0.005, 
                                'lambda_v': 0.005,
                                'alpha': 0.25, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(16, 43, 58),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001])
                                }, 
                            'Merge':{
                                'r': 1,
                                's': 1, 
                                'p': 0,
                                'f': 1, 
                                'm': 1, 
                                'tau': torch.Tensor([1]).long(), 
                                'v_th': 0.001,  
                                'lambda_X': 0.005, 
                                'lambda_v': 0.005,
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
                                'lambda_v': 0.015, 
                                'lambda_X': 0.015,
                                'alpha': 0.25, 
                                'beta': 0.5, 
                                'lambda_vth': 0.05, 
                                'alpha_vth': 20, 
                                'vth_rest': 0.1,
                                'vth_conv_params': [(64, 20, 27),1,(0,0,0)], #16, 43, 58, 1, 0
                                'stiffness_goal_inf': torch.as_tensor(1), 
                                'len_stiffness_buffer': torch.as_tensor(20),
                                'target_parameters': torch.tensor([12.327922591384661, 7.516300463119961e-05, 1.796295780751845]),
                                'v_th_gains': torch.tensor([0.0, 0.01, 0.000001]), 
                                'tau_max_gain': torch.as_tensor(1.5)
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
                                    - Input.f (int): number of maps in Input layer [-]
                                    - Input.m (int): number of synapses between any two neurons in Input layer [-]
                                    - Input.in (int): number of channels fed into input layer [-]
                                    
                                    - SSConv.r (int): kenerl size of SSConv layer [-]
                                    - SSConv.s (int): stride for SSConv layer [-]
                                    - SSConv.p (int): padding for SSConv layer [-]
                                    - SSConv.w_init (float): initial value of weights for training in SSConv layer[-]
                                    - SSConv.f (int): number of maps in SSConv layer [-]
                                    - SSConv.m (int): number of synapses between any two neurons in SSConv layer [-]
                                    - SSConv.tau (torch.tensor long): synaptic delay of neurons in SSConv layer [ms]
                                    - SSConv.v_th (float): voltage threshold for spiking of neurons in SSConv layer [-]
                                    - SSConv.lambda_X (float): time constant of presynaptic trace in SSConv layer [s]
                                    - SSConv.lambda_X (float): time constant of membrane potential in SSConv layer [s]
                                    - SSConv.alpha (float): scaling factor of presynaptic trace in SSConv layer [-]
                                    - SSConv.lambda_vth (float): time constant of the decay during the voltage threshold update in the SSConv layer [s]
                                    - SSConv.alpha_vth (float): scaling factor of voltage threshold update rule in SSConv layer [-]
                                    - SSConv.vth_rest (float): resting value of voltage threshold in SSConv layer [-]
                                    - SSConv.vth_conv_params (list): convolutional parameters defining window size for voltage threshold update in SSConv layer [-]
                                    - SSConv_stiffness_goal_inf (torch.tensor): target for the stiffness value during inference in SSConv layer [-]
                                    - SSConv_len_stiffness_buffer (torch.tensor): length of the temporal stiffness buffer in SSConv layer [-]
                                    - SSConv_target_parameters (torch.tensor): parameters specifying the target exponential stiffness function during training in SSConv layer [-]
                                    - SSConv_v_th_gains (torch.tensor): gains for voltage threshold update in SSConv layer [-]
                                
                         
                                    - Merge.r (int): kenerl size of Merge layer [-]
                                    - Merge.s (int): stride for Merge layer [-]
                                    - Merge.p (int): padding for Merge layer [-]
                                    - Merge.f (int): number of maps in Merge layer [-]
                                    - Merge.m (int): number of synapses between any two neurons in Merge layer [-]
                                    - Merge.tau (torch.tensor long): synaptic delay of neurons in Merge layer [ms]
                                    - Mergev.v_th (float): voltage threshold for spiking of neurons in Merge layer [-]
                            
                                    
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
                                    - MSConv.lambda_X (float): time constant of presynaptic trace in MSConv layer [s]
                                    - MSConv.lambda_X (float): time constant of membrane potential in MSConv layer [s]
                                    - MSConv.alpha (float): scaling factor of presynaptic trace in MSConv layer [-]
                                    - MSConv.lambda_vth (float): time constant of the decay during the voltage threshold update in the MSConv layer [s]
                                    - MSConv.alpha_vth (float): scaling factor of voltage threshold update rule in MSConv layer [-]
                                    - MSConv.beta (float): weight of inhibitory synaptic efficacies
                                    - MSConv.vth_rest (float): resting value of voltage threshold in MSConv layer [-]
                                    - MSConv.vth_conv_params (list): convolutional parameters defining window size for voltage threshold update in MSConv layer [-]
                                    - MSConv_stiffness_goal_inf (torch.tensor): target for the stiffness value during inference in MSConv layer [-]
                                    - MSConv_len_stiffness_buffer (torch.tensor): length of the temporal stiffness buffer in MSConv layer [-]
                                    - MSConv_target_parameters (torch.tensor): parameters specifying the target exponential stiffness function during training in MSConv layer [-]
                                    - MSConv_v_th_gains (torch.tensor): gains for voltage threshold update in MSConv layer [-]
                                    
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

            self.par_name = parameters['par_name']                                         

            self.directory = parameters['directory']                                        

            self.height = parameters['height']                                             
            self.width = parameters['width']                                               

            self.delta_ref = parameters['delta_ref']                                       

            ##Input layer
            self.input_in_dim = parameters['Input']['in']                                  
            self.input_out_dim = parameters['Input']['f']                              
            self.input_m = parameters['Input']['m']                                        
            self.input_kernel = parameters['Input']['r']                                   
            self.input_stride = parameters['Input']['s']                                    
            self.input_padding = parameters['Input']['p']                                   
            
            #Only relevant if multisynaptic connections with input 
            self.input_kernel3D = (self.input_m, self.input_kernel, self.input_kernel)      
            self.input_stride3D = (self.input_m, self.input_stride, self.input_stride)      
            self.input_padding3D = (0, self.input_padding, self.input_padding)             
            self.input_weights = torch.ones((self.input_out_dim, self.input_in_dim, self.input_kernel, self.input_kernel)) 

            ##SSConv layer: 
            self.SSConv_in_dim = self.input_out_dim                                          
            self.SSConv_out_dim = parameters['SSConv']['f']                                
            self.SSConv_kernel =  parameters['SSConv']['r']                                
            self.SSConv_stride =  parameters['SSConv']['s']                                
            self.SSConv_padding = parameters['SSConv']['p']                                
            self.SSConv_w_init = parameters['SSConv']['w_init']                            
            self.SSConv_m = parameters['SSConv']['m']                                                               
            self.SSConv_lambda_v = parameters['SSConv']['lambda_v']                         
            self.SSConv_lambda_X = parameters['SSConv']['lambda_X']                           
            self.SSConv_v_th =  parameters['SSConv']['v_th']                                
            self.SSConv_alpha = parameters['SSConv']['alpha']                                
            self.SSConv_delay = parameters['SSConv']['tau']                                 

            #Adaptive neuron parameters inference
            self.SSConv_lambda_vth = parameters['SSConv']['lambda_vth']                     
            self.SSConv_alpha_vth = parameters['SSConv']['alpha_vth']                       
            self.SSConv_vth_rest = parameters['SSConv']['vth_rest']                         
            self.SSConv_vth_conv_params =  parameters['SSConv']['vth_conv_params']          
            self.SSConv_stiffness_goal_inf = parameters['SSConv']['stiffness_goal_inf']
            #self.SSConv_N_sm = 9./(self.SSConv_vth_conv_params[0]**2)  
            self.SSConv_N_sm = 1                                                            
           

            #Adaptive neuron parameters training 
            self.SSConv_target_parameters = parameters['SSConv']['target_parameters']
            self.SSConv_len_stiffness_buffer = parameters['SSConv']['len_stiffness_buffer']
            self.SSConv_v_th_gains = parameters['SSConv']['v_th_gains'] 

            
            self.SSConv_kernel3D = (self.SSConv_m, self.SSConv_kernel, self.SSConv_kernel)  
            self.SSConv_stride3D = (self.SSConv_m, self.SSConv_stride, self.SSConv_stride)   
            self.SSConv_padding3D = (0, self.SSConv_padding, self.SSConv_padding)             
            self.SSConv_weight_shape = (self.SSConv_out_dim, self.SSConv_in_dim, self.SSConv_m, self.SSConv_kernel, self.SSConv_kernel) 
            self.SSConv_weights = self.SSConv_w_init*torch.ones((self.SSConv_out_dim, self.SSConv_in_dim, self.SSConv_m, self.SSConv_kernel, self.SSConv_kernel)) 
            self.SSConv_weights_pst = torch.ones((self.SSConv_out_dim, self.SSConv_in_dim, self.SSConv_m, self.SSConv_kernel, self.SSConv_kernel))           

            
            ##Merge layer 
            self.merge_in_dim = self.SSConv_out_dim                                         
            self.merge_out_dim = parameters['Merge']['f']                                    
            self.merge_kernel =  parameters['Merge']['r']                                    
            self.merge_stride =  parameters['Merge']['s']                                   
            self.merge_padding = parameters['Merge']['p']                                  
            self.merge_m = parameters['Merge']['m']                                         
            self.merge_v_th = parameters['Merge']['v_th']                                    
            self.merge_delay = parameters['Merge']['tau']  
            self.merge_lambda_v = parameters['Merge']['lambda_v']                         
            self.merge_lambda_X = parameters['Merge']['lambda_X']                                   
            
            self.merge_kernel3D = (self.merge_m, self.merge_kernel, self.merge_kernel)       
            self.merge_stride3D = (self.merge_m, self.merge_stride, self.merge_stride)      
            self.merge_padding3D = (0, self.merge_padding, self.merge_padding)                 
            self.merge_weights = torch.ones((self.merge_out_dim, self.merge_in_dim, self.merge_m, self.merge_kernel, self.merge_kernel))      
            self.merge_weights_pst = torch.ones((self.merge_out_dim, self.merge_in_dim, self.merge_m, self.merge_kernel, self.merge_kernel))  

            ##MSConv layer: modified LIF neurons 
            self.MSConv_in_dim = self.merge_out_dim                                         
            self.MSConv_out_dim = parameters['MSConv']['f']                                 
            self.MSConv_kernel =  parameters['MSConv']['r']                                 
            self.MSConv_stride =  parameters['MSConv']['s']                                 
            self.MSConv_padding = parameters['MSConv']['p']                                 
            self.MSConv_w_init_ex = parameters['MSConv']['w_init_ex']                      
            self.MSConv_w_init_inh = parameters['MSConv']['w_init_inh']                     
            self.MSConv_m = parameters['MSConv']['m']                                                                                     
            self.MSConv_lambda_X = parameters['MSConv']['lambda_X']   
            self.MSConv_lambda_v = parameters['MSConv']['lambda_v']                          
            self.MSConv_v_th = parameters['MSConv']['v_th']                                  
            self.MSConv_alpha = parameters['MSConv']['alpha']                                
            self.MSConv_beta = parameters['MSConv']['beta']                                  
            self.MsConv_tau_min = parameters['MSConv']['tau_min'] 
            self.MsConv_tau_max = parameters['MSConv']['tau_max'] 

            #Adaptive neuron parameters inference
            self.MSConv_lambda_vth = parameters['MSConv']['lambda_vth']                     
            self.MSConv_alpha_vth = parameters['MSConv']['alpha_vth']                          
            self.MSConv_vth_rest = parameters['MSConv']['vth_rest']                         
            self.MSConv_vth_conv_params =  parameters['MSConv']['vth_conv_params']
            self.MSConv_stiffness_goal_inf = parameters['MSConv']['stiffness_goal_inf']
            #self.SSConv_N_sm = 9./(self.SSConv_vth_conv_params[0]**2)  
            self.MSConv_N_sm = 1                                                            
            self.MSConv_tau_max_gain = parameters['MSConv']['tau_max_gain']
           

            #Adaptive neuron parameters training 
            self.MSConv_target_parameters = parameters['MSConv']['target_parameters']
            self.MSConv_len_stiffness_buffer = parameters['MSConv']['len_stiffness_buffer']
            self.MSConv_v_th_gains = parameters['MSConv']['v_th_gains'] 
            
            self.MSConv_delay = torch.linspace(self.MsConv_tau_min, self.MsConv_tau_max, self.MSConv_m).to(dtype=torch.long)  
            self.MSConv_kernel3D = (self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel)   
            self.MSConv_stride3D = (self.MSConv_m, self.MSConv_stride, self.MSConv_stride)   
            self.MSConv_padding3D = (0, self.MSConv_padding, self.MSConv_padding)               
            self.MSConv_weight_shape = (self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel)  
            self.MSConv_weights_ex = self.MSConv_w_init_ex*torch.ones((self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel))    
            self.MSConv_weights_inh = self.MSConv_w_init_inh*torch.ones((self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel))  
            self.MSConv_weights_pst = torch.ones((self.MSConv_out_dim, self.MSConv_in_dim, self.MSConv_m, self.MSConv_kernel, self.MSConv_kernel))  

            
            ##Pooling layer
            self.pooling_in_dim = self.MSConv_out_dim                                        
            self.pooling_out_dim =  parameters['Pooling']['f']                               
            self.pooling_kernel =  parameters['Pooling']['r']                              
            self.pooling_stride =  parameters['Pooling']['s']                              
            self.pooling_padding = parameters['Pooling']['p']                               
            self.pooling_m = parameters['Pooling']['m']                                    
            self.pooling_lambda_i = parameters['Pooling']['lambda']                         
            self.pooling_lambda_v = parameters['Pooling']['lambda']                          
            self.pooling_lambda_X = parameters['Pooling']['lambda']                             
            self.pooling_v_th = parameters['Pooling']['v_th']                               
            self.pooling_delay = parameters['Pooling']['tau']                               
            
            self.pooling_kernel3D = (self.pooling_m, self.pooling_kernel, self.pooling_kernel)   
            self.pooling_stride3D = (self.pooling_m, self.pooling_stride, self.pooling_stride)    
            self.pooling_padding3D = (0, self.pooling_padding, self.pooling_padding)                
            self.pooling_weights = torch.ones((self.pooling_out_dim, self.pooling_in_dim, self.pooling_m, self.pooling_kernel, self.pooling_kernel))      
            self.pooling_weights_pst = torch.ones((self.pooling_out_dim, self.pooling_in_dim, self.pooling_m, self.pooling_kernel, self.pooling_kernel))  



            ##Dense layer 
            self.dense_in_dim = self.pooling_out_dim                                       
            self.dense_out_dim = parameters['Dense']['f']                                   
            self.dense_w_init = parameters['Dense']['w_init']                               
            self.dense_m = parameters['Dense']['m']                                            
            self.dense_lambda_i = parameters['Dense']['lambda']                             
            self.dense_lambda_v = parameters['Dense']['lambda']                             
            self.dense_lambda_X = parameters['Dense']['lambda']                                 
            self.dense_v_th = parameters['Dense']['v_th']                                   
            self.dense_alpha = parameters['Dense']['alpha']                                
            self.dense_delay = parameters['Dense']['tau']                                   
            
            self.dense_stride3d = (1,1,1)                                                     
            self.dense_padding3d = (0,0,0)
            #dense_weights = SSConv_w_init*torch.ones((SSConv_out_dim, SSConv_in_dim, SSConv_kernel, SSConv_kernel)) #Inital weight tensor
            #dense_weights_pst = torch.ones((SSConv_out_dim, SSConv_in_dim, SSConv_kernel, SSConv_kernel))           #Weights for computation of presynaptic trace


            ##Training parameters 
            self.training_n = parameters['training']['n']                                  
            self.training_a = parameters['training']['a']                                   
            self.training_L= parameters['training']['L']                                     
            self.training_ma_len = parameters['training']['ma_len']                         


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
                        (0, 140, 255),     #dark orange
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
    
        elif self.data == 'checkerboard':
            return self.define_SNN_param(self.data_checkerboard)
        
        elif self.data == 'rot_disk':
            return self.define_SNN_param(self.data_rot_disk)
        
        elif self.data == 'roadmap':
            return self.define_SNN_param(self.data_roadmap)

        elif self.data == 'ODA_dataset':
            return self.define_SNN_param(self.data_ODA_dataset)
        
        else: 
            print('Invalid selection')
        



