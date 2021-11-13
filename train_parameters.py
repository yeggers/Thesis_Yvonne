import random
import os
import pickle
import pylab as pl

import torch
from deap import base, creator, tools
import numpy as np 
from models.generations import Radar
import networkx

from models.SNN_Federico2_noSSConv import SNN_Federico_noSSConv
from SNN_param import SNN_param
from SNN_param_noSSConv import SNN_param_noSSConv
from train_MSConv_noSSConv import main_MSConv_noSSConv
from tools.OF_vectors import compute_OF
from models.layer_definitions_noSSConv import MSConv
import matplotlib.pyplot as plt

def initial_population(alpha_range, lambda_range, vth_range, tau_max_range):
    """Create educated guess for initial population by supplying ranges within which the parameters are expected to be.
    
    Parameters: 
        alpha_range (np.array): expected range of alpha 
        lambda_range (np.array): expected range of lambda
        vth_range (np.array): expected range of voltage threshold
        tau_max_range (np.array): expected range of maximum delay
    """

    #MSConv_alpha_training, MSConv_lambda_training, MSConv_vth_training, MSConv_alpha_inference, MSConv_lambda_inference, MSConv_vth_inference, tau_max]

    individual = creator.Individual([np.random.choice(alpha_range), np.random.choice(lambda_range), np.random.choice(vth_range), np.random.choice(alpha_range), np.random.choice(lambda_range), np.random.choice(vth_range), np.random.choice(tau_max_range)])

    return individual


    


def output_SNN_Federico(parameters, batch_size, iterations, new_weights, weights_name_exc, weights_name_inh, device, dt, individual):
    """Compute OF with Federico's SNN
    
    Parameters: 
        parameters (SNN_param_noSSConv): parameters of the SNN
        batch_size (int): batch size used during training
        iterations (int): number of iterations used during training 
        new_weights (bool): bool determining whether or not new weights should be created 
        weights_name_exc (string): name of excitatory weights 
        weights_name_exc (string): name of inhibitory weights 
        device (torch.device): device on which computations shall be performed
        dt (float): time step of the simulation
        individual (creator.individual): 

    """
    # =====================================================================================
    # Training
    # =====================================================================================
    
    #Use parameters of individuals for training
    parameters.MSConv_alpha = individual[0]
    parameters.MSConv_lambda_i = individual[1]
    parameters.MSConv_lambda_v = individual[1]
    parameters.MSConv_lambda_X = individual[1]
    parameters.MSConv_v_th = individual[2]
    parameters.MsConv_tau_max = individual[6]

    #Train MSConv layer 
    s_STDP_exc, s_STDP_inh = main_MSConv_noSSConv(dt, batch_size, iterations, new_weights, par_name, parameters, weights_name_exc, weights_name_inh, device)
    
    # =====================================================================================
    # Inference
    # =====================================================================================
    
    #Use parameters of individuals for inference 
    parameters.MSConv_alpha = individual[3]
    parameters.MSConv_lambda_i = individual[4]
    parameters.MSConv_lambda_v = individual[4]
    parameters.MSConv_lambda_X = individual[4]
    parameters.MSConv_v_th = individual[5]

    #Randomly selecting sequence from directory 
    random_files = random.sample(os.listdir(parameters.directory), 1)
    
    #Loading first sequence
    data = torch.load(parameters.directory + '/{}'.format(random_files[0])).to(device)

    #Determine sequence length
    seq_length = data.shape[0]

    #Specify desired name of initial excitatory weights or name of the existing weights 
    weights_name_exc = 'weights/MSConv/{par}/noSSConv/{weights_name}'.format(par = par_name, weights_name = weights_name_exc)

    #Specify desired name of initial excitatory weights or name of the existing weights 
    weights_name_inh = 'weights/MSConv/{par}/noSSConv/{weights_name}'.format(par = par_name, weights_name = weights_name_inh)
    
    #Creating instance of network
    SNN = SNN_Federico_noSSConv(weights_name_exc, weights_name_inh, parameters, device, 1, dt)

    #Retrieve OF vectors corresponding to various maps and expand to match size of output spikes
    output_height = MSConv(parameters, dt).conv_MSConv_dim[0]
    output_width = MSConv(parameters, dt).conv_MSConv_dim[1]
    OF_class = compute_OF(parameters, s_STDP_exc.weights, s_STDP_inh.weights, device, -1)
    OF_vectors = torch.tensor(OF_class.compute_OF()[-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(1, -1, 1, output_height, output_width, -1).to(device)
  
    #Set up OF vector 
    OF = torch.zeros(seq_length, output_height, output_width, 2).to(device)
    
    for ts in range(seq_length):
        #Expand z to contain x- and y-direction of OF 
        z = SNN.forward(data[ts]).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 2) 
        #Compute OF vector at each pixel according to OF vectros for each map
        z = z*OF_vectors
        #Flatten output spikes to 2 dimension 
        z = torch.mean(z, dim = (0,1,2))
        #Save OF for current time step 
        OF[ts] = z

    return OF, random_files



def ground_truth(parameters, sequence, device):
    """Compute ground truth OF
    
    Parameters: 
        sequence (string): directory of sequence 
        parameters(SNN_param): parameters of the SNN
        device (torch.device): device on which computations shall be performed 

    """

    ground_truth = torch.load(parameters.directory_gt + '/{}'.format(sequence[0])).to(device)

    #Compute OF velocity 
    OF_vel = (parameters.MSConv_kernel - (-parameters.MSConv_kernel))/(parameters.MSConv_kernel-1)/(parameters.MsConv_tau_max - parameters.MsConv_tau_min)

    #Multiply ground truth with OF velocity 
    ground_truth = OF_vel * ground_truth
    
    #Permute tensor so it matches shape (N, Ch, h, w) for dowmsampling
    ground_truth = ground_truth.permute(0, 3, 1, 2)

    #Downsample to match output size after input layer 
    ground_truth = torch.nn.MaxPool2d(parameters.input_kernel, parameters.input_stride, parameters.input_padding)(ground_truth)

    #Downsample to match output size after MSConv layer 
    ground_truth = torch.nn.MaxPool2d(parameters.MSConv_kernel, parameters.MSConv_stride, parameters.MSConv_padding)(ground_truth)

    #Permute back to original shape 
    ground_truth = ground_truth.permute(0, 2, 3, 1)

   
    return ground_truth

    #Compute OF for the sequence for which MSConv inference was performed 

def evaluate(parameters, batch_size, iterations, new_weights, weights_name_exc, weights_name_inh, device, dt, individual):
    """Evaluate the performance of the individuals
    
    Parameters:
        parameters (SNN_param_noSSConv): parameters of the SNN
        batch_size (int): batch size used during training
        iterations (int): number of iterations used during training 
        new_weights (bool): bool determining whether or not new weights should be created 
        weights_name_exc (string): name of excitatory weights 
        weights_name_exc (string): name of inhibitory weights 
        device (torch.device): device on which computations shall be performed
        dt (float): time step of the simulation
        individual (creator.individual): 
    """

    #Compute optical flow with Federicos SNN
    OF, random_files = output_SNN_Federico(parameters, batch_size, iterations, new_weights, weights_name_exc, weights_name_inh, device, dt, individual)

    
    #Determine ground truth 
    OF_ground_truth = ground_truth(parameters, random_files, device)
    

    #Compute error
    evaluation = ((torch.sum((OF - OF_ground_truth)**2))**0.5).to('cpu')


    return evaluation,

def main(n, CXPB, MUTPB, NGEN, hof_maxsize, checkpoint = None):
    """Evaluate the performance of the individuals
    
    Parameters:
       n (int): number of individuals in population 
       CXPB (float): crossover probability
       MUTPB (float): mutation probability
       NGEN (float): number of generations
       hof_maxsize (int): maximum size of hall of fame
       checkpoint (str): name of checkpoint file 
    """
    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
        pop = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        history = cp["history"]
    else:
        # Start a new evolution
        start_gen = 1
        halloffame = tools.HallOfFame(maxsize=hof_maxsize)
        logbook = tools.Logbook()
        history = tools.History()
        pop = toolbox.population(n)
        history.update(pop)

    
    #Make statistics 
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)
   
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    i = 1
    fitness_collect = []
    for ind, fit in zip(pop, fitnesses):
        print(" ")
        print('Evaluating individual number: ', i)
        print(" ")
        ind.fitness.values = fit 
        fitness_collect.append(fit)
        i += 1
       
    record = stats.compile(pop)
    logbook.record(gen = 0, evals = len(pop), **record)
    
    print(logbook)

    cp = dict(population=pop, generation = 0, halloffame=halloffame,
                logbook=logbook, fitnesses = fitness_collect, rndstate=random.getstate(), history = history)

    with open("logbooks/checkpoint_0.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)
   
   
    for g in range(start_gen, NGEN):
        
        print('Generation: ', g)
        print(" ")
       
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
     
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        invalid_ind_idx = [i for i in range(len(offspring)) if not offspring[i].fitness.valid]
        
        fitnesses = map(toolbox.evaluate, invalid_ind)
        i = 0
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            fitness_collect[invalid_ind_idx[i]] = fit
            print('Evaluating individual number: ', invalid_ind_idx[i])
            i += 1

        # The population is entirely replaced by the offspring
        pop[:] = offspring

       
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen = g, evals = len(invalid_ind), **record)
        
        print(logbook)
        
        # Fill the dictionary using the dict(key=value[, ...]) constructor
        cp = dict(population=pop, generation = g, halloffame=halloffame,
                    logbook=logbook, fitnesses = fitness_collect, rndstate=random.getstate(), history = history)

        test = "logbooks/checkpoint_{gen}.pkl".format(gen = g)

        with open("logbooks/checkpoint_{gen}.pkl".format(gen = g), "wb") as cp_file:
            pickle.dump(cp, cp_file)
            
    return pop





if __name__ == "__main__":


    #Initialize parameters 
    par_name = 'lines'
    parameters = SNN_param_noSSConv(par_name).define_data()

    #Simulation timestep 
    dt = 10**(-3)

    #Batch size (Not that only one sequenze is available for the rotating disk and the checkerbord)
    batch_size = 1

    #Number of iterations to perform during training
    iterations = 3

    #Specify whether to initialize new weights or to use existing ones 
    new_weights = True

    #TODO: check if there is a way to not overwrite the weights everytime 
    #Name of weights 
    weights_name_exc = 'MSConvWeights_lines_exc_test1.pt'
    weights_name_inh = 'MSConvWeights_lines_inh_test1.pt'

    #Define number of parameters to be trained 
    #Parameters are [MSConv_alpha_training, MSConv_lambda_training, MSConv_vth_training, MSConv_alpha_inference, MSConv_lambda_inference, MSConv_vth_inference, tau_max]
    IND_SIZE = 7

    #Number of individuala in populations
    n = 2

    #Crossover probability 
    CXPB = 0.1

    #Mutation parameter
    MUTPB = 0.5

    #Number of generations
    NGEN = 3

    #Number of individuals in hof
    hof_maxsize = 1

    #Plotting 
    plotting = False

    #Ranges for initial guess
    alpha_step = 0.09 
    lambda_step = 0.003
    vth_step = 0.15
    tau_max_step = 1
    
    alpha_range = np.arange(0.075, 0.45, alpha_step)
    lambda_range = np.arange(0.002, 0.009, lambda_step)
    vth_range = np.arange(0.1, 5, vth_step)
    tau_max_range = np.arange(5, 15, tau_max_step)
    
    #Mutation paramters 
    sigma = [alpha_step, lambda_step, vth_step, alpha_step, lambda_step, vth_step, tau_max_step]

    #Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
        print("Running on the GPU")
        print(" ")

    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        print(" ")

    #Create types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    #Make toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", initial_population, alpha_range, lambda_range, vth_range, tau_max_range)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate, parameters, batch_size, iterations, new_weights, weights_name_exc, weights_name_inh, device, dt)



    # #Perform EA
    # population = main(n, CXPB, MUTPB, NGEN, hof_maxsize)

    with open('logbooks/logbook1.pickle', 'rb') as f:
            pop = pickle.load(f)


    if plotting:

        graph = networkx.DiGraph(history.genealogy_tree)
        graph = graph.reverse()     # Make the graph top-down
        colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
        networkx.draw(graph, node_color=colors)
        plt.show()
        
        with open('logbook.pickle', 'rb') as f:
            pop = pickle.load(f)

        num_steps = 10
        minima = np.array([0.225, 0.001, 0, 0.225, 0.001, 0, 5])
        maxima = np.array([0.725, 0.008, 7, 0.725, 0.008 , 7, 15])
        steps = (maxima - minima)/num_steps

        titles = ['alpha_training', 'lambda_training', 'vth_training', 'alpha_inference', 'lambda_inference', 'vth_inferece', 'tau_max']
        labels = [np.round(np.linspace(minima[0], maxima[0], num_steps), 3), np.round(np.linspace(minima[1], maxima[1], num_steps), 4), np.round(np.linspace(minima[2], maxima[2], num_steps), 1), np.round(np.linspace(minima[3], maxima[3], num_steps), 3), np.round(np.linspace(minima[4], maxima[4], num_steps), 4), np.round(np.linspace(minima[5], maxima[5], num_steps), 1), np.round(np.linspace(minima[6], maxima[6], num_steps), 0)
        ]

        for generation in range(len(pop)):
            fig = pl.figure(figsize=(6, 6))
            radar = Radar(fig, titles, labels, num_steps)
            i = 0
            for individual in pop[generation]['pop']:
                
                radar.plot((individual - minima)/steps + 1,  "-", lw=2, alpha=0.4, label= 'individual_{}'.format(i))
                i +=1
            radar.ax.legend()

            plt.savefig('star_plot_gen{generation}'.format(generation = generation))
        plt.show()


 
