

import math
import os
import bz2
import pickle
import _pickle as cPickle

import aedat
import bagpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from bagpy import bagreader
from mpl_toolkits import mplot3d
from norse.torch.functional.lif import (LIFParameters, lif_current_encoder,
                                        lif_feed_forward_step, lif_step)

import tools.odd_even_fix as fix


class import_DVS:
    '''This class converts the event stream of an event camera into a tensor containing discrete time bins
    
    Parameters: 
        height (int): number of pixels along the vertical direction in the input image
        width (int): number of pixels along the horizontal direction in the input image 
    '''

    def __init__(self, height = 264, width = 320): 
        
        #Image height 
        self.height = height 
        
        #Image width 
        self.width = width


        
    def read_bag_data(self, dir):
        '''This function reads events in bag format into a list of frames containing the events.
        
        Parameters:
            dir (str): directory of the bag file 
        '''
        #Reading data
        b = bagreader(dir)
        
        #Showing topic table to see which topic to choose 
        print(b.topic_table)
        
        #Selecting desired topic 
        newmsg = b.message_by_topic(topic = 'raw_data')

        #Reading resulting csv file 
        newmsg = pd.read_csv(self.dir)
       
        #Disregarding false entries 
        newmsg = [frame for frame in newmsg if frame.events[0].ts.secs !=0]
        
        return newmsg

    def read_csv_data(self, dir, fix_stripes = False):
        '''This function reads events in csv format into a panda data frame containing the events.
        
        Parameters:
            dir (str): directory of the csv file 
        '''

        #Reading csv file 
        newmsg = pd.read_csv(dir, header = 0, names= ['t', 'x', 'y', 'pol'])


        #Apply function to fix stripes in data 
        if fix_stripes:
            #Convert data frame to dictionary 
            data = newmsg.to_dict('list')
            #Convert dictionary entries to numpy arrays 
            data = dict([key, np.array(data[key])] for key in ["x", "y", "t", "pol"])
            test = data['x']
            #Applying odd_even fix 
            data = fix.odd_even_fix(data)
            #Converting back to data frame 
            #data = pd.DataFrame.from_dict(data)

        return newmsg

    def read_aedat_data(self, dir):
        '''This function reads events in aedat format into a list of events.
        
        Parameters:
            dir (str): directory of the bag file 
        '''

        #Reading aedat file 
        decoder = aedat.Decoder(dir)

        #Extracting events and putting them in a list 
        data = [events for packet in decoder if 'events' in packet for events in packet['events']]

        return data


    def event_arrays(self, dir, start_frame, end_frame):
        '''This function distributes the events contained in the frames of an event list created from a bag file into a tensor with discrete time bins. 
        
        Parameters: 
            dir (str): directory containing the bag file 
            start_frame (int): first frame to be included 
            end_frame (int):last frame to be included
        '''
        
        #Getting data 
        newmsg = self.read_bag_data(dir)
        
        #Setting up arrays to collect events 
        X = []
        Y = []
        t = []
        polarity = []
        Yo = []
        
        #Initializing counter for when more than one frame is considered 
        counter = 0 

        for frame in range(start_frame, end_frame + 1, 1):
            #Adding entries to collect all events within frame
            X = np.hstack((X, np.zeros(len(newmsg[frame].events))))
            Yo = np.hstack((Yo, np.zeros(len(newmsg[frame].events))))
            Y = np.hstack((Y, np.zeros(len(newmsg[frame].events))))
            t =  np.hstack((t, np.zeros(len(newmsg[frame].events))))
            polarity = np.hstack((polarity, np.zeros(len(newmsg[frame].events))))

            for event in range(len(newmsg[frame].events)):
                #Collecting data from all events within specified frames
                Yo[counter] = newmsg[frame].events[event].x
                X[counter] = self.width - 1 - newmsg[frame].events[event].x
                Y[counter] = self.height - 1 - newmsg[frame].events[event].y
                t[counter] = newmsg[frame].events[event].ts.nsecs
                polarity[counter] = newmsg[frame].events[event].polarity
                counter += 1
     
        return X, Y, t, polarity 

    def group_on_off(self, dir, start_frame, end_frame):
        '''This function divides the events contained in a tensor obtained from a bag file into on and off events. 
        
        Parameters: 
            dir (str): directory containing the bag file 
            start_frame (int): first frame to be included 
            end_frame (int):last frame to be included
        '''
        #Dividing data into on and off events
        X, Y, t, polarity = self.event_arrays(dir, start_frame, end_frame)
        data = pd.DataFrame({'Time': t, 'X-pixel': X , 'Y-pixel': Y, 'On/off': polarity})
        groups = data.groupby('On/off')
        
        return groups 


    def get_csv_input(self, dir, dir_result, dt = 10**(-3), dt_data = 10**(-6), fix_stripes = False, data_flipped = False, start = 0):
        '''This function converts the panda data frane obtained from a csv file into a tensor with diescrete timebins containing the events. 
        
        Parameters: 
            dir (str): directory containing the csv file 
            dir_result (str): name of directory in which result shall be saved
            dt (float): size of time bins (seconds)
            dt_data (float): time step in event data
            fix_stripes (bool): specifies whether stripe fix shall be applied
            data_flipped (bool): specifies whether polarity is given denoted by [-1,1] instead of [0,1]
            start (int): only considers 
        '''
        
        #Read image data 
        data = self.read_csv_data(dir, fix_stripes)
  
        #Determine time of first event 
        offset = data.t[0 + start] 
        end = data.t.iloc[-1] 

        #Determine length of data
        length = np.ceil((end - offset)*(dt_data)/dt) + 1 


        #Initializing tensor with correct dimensions(time [ms] x 1 x number of channels x pixel height x pixel width)
        Input = torch.zeros((int(length), 1, 2, self.height, self.width))
        
        if data_flipped:
            #Converting -1, 1 polarity to 0,1 values 
            data.pol = round((data.pol+1)/2)
            #Reading out data
        
        for i in range(start, len(data.t)):
            #Finding index of event
            idx = np.floor((data.t[i] - offset)*(dt_data)/dt)
            Input[int(idx), 0, int(data.pol[i]), self.height - 1 - int(data.y[i]), self.width - 1 - int(data.x[i])] = 1
                
        
        
        # #Only keeping odd pixels 
        # even_x = np.arange(1, Input.shape[4]+1, 2)
        # even_y = np.arange(1, Input.shape[3]+1, 2)
        # Input = Input[:, :, :, 1::2, 1::2]

        with bz2.BZ2File(dir_result, 'w') as f: 
            cPickle.dump(Input, f)
            
        
        # #Saving tensor 
        # torch.save(Input, directory_name)

        return Input



    def get_aedat_input(self, dir, dir_result, dt = 10**(-3), dt_data = 10**(-6)):
        '''This function converts the list of events obtained from an aedat file into a tensor with diescrete timebins containing the events. 
        
        Parameters: 
            dir (str): directory containing the bag file 
            dt (float): size of time bins (seconds)
        '''

        #Reading events from aedat4 file 
        data = self.read_aedat_data(dir)
        
        #Splitting in 2, to save memory 
        data = data[:int(len(data)/2)]

        #Determine time of first event 
        offset = data[0][0]
        
        #Determine length of data
        length = np.ceil((data[-1][0] - offset)*dt_data/dt) + 1
        
        #Initializing tensor with correct dimensions(one row for each simulated time step containing one row each for on and off event. Those rows have the dimensions of the image.)
        Input = torch.zeros((int(length), 1, 2, self.height, self.width))
        
        #Reading out data
        for event in data:
            #Finding index of event
            idx = round((event[0] - offset)*dt_data/dt)
            Input[int(idx), 0, int(event[3]), event[2], event[1]] = 1

        # even_x = np.arange(0, Input.shape[4], 2)
        # even_y = np.arange(0, Input.shape[3], 2)
        # Input = Input[:, :, :, ::2, tuple(even_x)]

        with bz2.BZ2File(dir_result, 'w') as f: 
            cPickle.dump(Input, f)

        #torch.save(Input, dir_result)

        return Input

     
    def get_bag_input(self, dir,  dt = 3*10**(-3)):
        '''This function converts the panda data frane obtained from a bag file into a tensor with diescrete timebins containing the events. 
        
        Parameters: 
            dir (str): directory containing the bag file 
            dt (float): size of time bins (seconds)
        '''
        
        #Read image data 
        data = self.read_data(dir)
        
        #Determine time of first event 
        offset = data[0].events[0].ts.secs + data[0].events[0].ts.nsecs/10**9
        
        #Determine length of data
        length = np.ceil(((data[-1].events[-1].ts.secs + data[-1].events[-1].ts.nsecs/10**9) - offset)/dt)
        
        #Initializing tensor with correct dimensions(one row for each simulated time step containing one row each for on and off event. Those rows have the dimensions of the image.)
        Input = torch.zeros((int(length), 1, 2, self.height, self.width))
        
        #Reading out data
        for frame in data:
            for event in frame.events:
                #Finding index of event
                idx = np.floor((event.ts.secs + event.ts.nsecs/10**9 - offset)/dt)
                Input[int(idx), 0, int(event.polarity), self.height - 1 - event.y, self.width - 1 - event.x] = 1
        
        return Input
    
    def make_checkerboard_data(self, dir, dir_result, dt = 10**(-3), dt_data = 10**(-6)):
        '''This function creates tensors containing the events for all checkerboard sequences (aedat files). 
        
        Parameters:
            dir (str): directory containing the aedat files
            dir_results (str): directory in which tensors shall be saved
            dt (float): simulation time step
            dt_data (float): times step of the data
        '''
        
        folder_names = ['checkerboard', 'horizontal', 'vertical']
        folder_names = ['vertical']

        for folder in folder_names:
            print("Creating {} sequence".format(folder))
            dir_sequence = dir + '/{}.aedat4'.format(folder)
            dir_name = dir_result + '/{}.pbz2'.format(folder)
            data = self.get_aedat_input(dir_sequence, dir_name, dt = dt, dt_data = dt_data)

    
    def make_rotating_disk_data(self, dir, dir_result, fix_stripes = False, dt = 10**(-3), dt_data = 10**(-6), data_flipped = False):
        '''This function creates tensors containing the events for the rotating disk sequence and additionally creates a cropped and rotated version (csv files). 
        
        Parameters:
            dir (str): directory containing the aedat files
            dir_results (str): directory in which tensors shall be saved
            dt (float): simulation time step
            dt_data (float): times step of the data
        '''

        print('Creating unaltered sequence')
        data = self.get_csv_input(dir, dir_result + '/rot_disk.pbz2', fix_stripes=fix_stripes, dt = dt, dt_data = dt_data, data_flipped=data_flipped)

        #Crop sequence to square window
        print('Creating cropped sequence')
        data_cropped = torchvision.transforms.CenterCrop(self.height)(data)
        with bz2.BZ2File(dir_result + '/rot_disk_cropped.pbz2', 'w') as f: 
            cPickle.dump(data_cropped, f)

        #Rotate sequence by +90 degrees
        print('Creating sequence rotated +90 degrees')
        data_rotated_plus = torch.rot90(data_cropped, (1), (3,4))
        with bz2.BZ2File(dir_result + '/rot_disk_rotated_plus90.pbz2', 'w') as f: 
            cPickle.dump(data_rotated_plus, f)

        #Rotate sequence by -90 degrees
        print('Creating sequence rotated -90 degrees')
        data_rotated_minus = torch.rot90(data_cropped, (-1), (3,4))
        with bz2.BZ2File(dir_result + '/rot_disk_rotated_minus90.pbz2', 'w') as f: 
            cPickle.dump(data_rotated_minus, f)

            

    def make_roadmap_data(self, dir, dir_result, fix_stripes = False, dt = 10**(-3), dt_data = 10**(-6), data_flipped = False):
        '''This function creates tensors containing the events for all roadmap sequences (csv files). 
        
       Parameters:
            dir (str): directory containing the aedat files
            dir_results (str): directory in which tensors shall be saved
            dt (float): simulation time step
            dt_data (float): times step of the data
        '''
        #Counting number of data sets in directory 
        #TODO: Find more elegant way of defining number of sequences (Not hardcoded)
        N = 51
        sequence_number = np.arange(N)

        #Creating tensors containing sequence data 
        for sequence in sequence_number:
            print("Creating sequence {}".format(sequence))
            dir_sequence = dir + '/final/final_{}/events.csv'.format(sequence)
            dir_name = dir_result + '/roadmap{}.pbz2'.format(sequence)
            data = self.get_csv_input(dir_sequence, dir_name, fix_stripes=fix_stripes, dt = dt, dt_data = dt_data, data_flipped=data_flipped)
            

           
        
    def make_ODA_data(self, dir, dir_result, fix_stripes = False, dt = 10**(-3), dt_data = 10**(-9), data_flipped = True):
        '''This function creates tensors containing the events for all sequences in the ODA dataset (csv files). 
        
        Parameters:
            dir (str): directory containing the aedat files
            dir_results (str): directory in which tensors shall be saved
            dt (float): simulation time step
            dt_data (float): times step of the data
        '''

        folder_names = np.array([3, 10, 345])

        for folder in folder_names:
            print("Creating sequence {}".format(folder))
            dir_sequence = dir + '/{}/dvs.csv'.format(folder)
            dir_name = dir_result + '/ODA_dataset{}.pt'.format(folder)
            data = self.get_csv_input(dir_sequence, dir_name, fix_stripes=fix_stripes, dt = dt, dt_data = dt_data, data_flipped=data_flipped, start = 200000)

    
            
       

           








