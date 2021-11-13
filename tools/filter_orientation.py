import torch 
from scipy.optimize import minimize
import numpy as np 
import cv2
import matplotlib.pyplot as plt


def compute_objective_function( parameters, X, Y, brightness, d_max):
    """This function computes the minimum distances between a set of points (x,y) and a line defined by ax + by + c = 0 
    Parameters: 
        parameters (np.array): a, b, and c parameters specifying line
        X (torch.tensor): x-coordinates of the points
        Y (torch.tensor): y-coordinates of the points 
        brightness (torch.tensor): brightness values of the points   
        d_max (float): maximum distance within filter
    """

    #Obj = -brightness*(d_max - (parameters[0]*X + parameters[1]*Y + parameters[2])/(parameters[0]**2 + parameters[1]**2)**0.5)
    Obj = - torch.sum(brightness*(d_max - abs((parameters[0]*X + parameters[1]*Y + parameters[2])/(parameters[0]**2 + parameters[1]**2)**0.5)))

    return Obj


def plot_line(parameters, X):
    """This function plots the line approximating the filter obtained by 
    Parameters: 
        parameters (np.array): a, b, and c parameters specifying line
        X (torch.tensor): x-coordinates of the points
    """
    slope = -parameters[0]/parameters[1]
    intercept = - parameters[2]/parameters[1]
    Y = slope*X + intercept

    plt.plot(X, Y)
    plt.show()

X0 = np.array([1,1,-5])
r = 5
d_max = (2*r**2)**0.5

brightness = torch.diag(torch.ones(5))
X = torch.arange(5).reshape(5,1).expand(-1, 5)
Y = r - 1 - torch.arange(5).expand(5, -1)


args = (X, Y, brightness, d_max)



parameters = minimize(compute_objective_function, X0, args,  options={'gtol': 100, 'disp': True})
print(parameters)

cv2.imshow('test', np.array(brightness))
cv2.waitKey(1)



test = compute_objective_function(parameters.x, X, Y, brightness, d_max)
tes2 = compute_objective_function(np.array([1, 1, -4]), X, Y, brightness, d_max)


plot_line(np.array(parameters.x, X[0]))




