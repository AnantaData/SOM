__author__ = 'Damith/laksheen'

import random
import numpy as np
import math
import sys

class SOM(object):

    '''
    Constructor Function: Takes as input the following parameters x_size : the width of the map desired
    y_size : the height of the  map desired input_dims : the dimensinality of the vectors the som is trained to deal with
    '''

    def __init__(self,x_size,y_size,input_dims):
        self.num_x=x_size
        self.num_y=y_size
        self.dimension=input_dims
        self.neurons=np.random.randint(2,size=(self.num_x,self.num_y,self.dimension))
        #self.neurons = np.ones(shape=(self.num_x,self.num_y,self.dimension))                #initialize to neurons with weight 1
        self.dissim = None
        self.radius=min(x_size,y_size)/2

    '''
    _get_min_dissimilarity calculate the dissimilarity measures (here the eculidean distance) to all the neurons for a given
    training/input vector and returns the closestly similar vector's coordinates as output
    '''
    def _get_min_dissimilarity(self,input_vector):
        self.dissim = np.zeros(shape=(self.num_x,self.num_y))
        for x in range(self.neurons.shape[0]):
            for y in range(self.neurons.shape[1]):
                self.dissim[x][y] = np.linalg.norm(self.neurons[x][y]-input_vector)
                print 'dissim ',x,y,' : ',self.dissim[x][y],
            print ' '
        max_fit=np.array([np.argmin(self.dissim)/self.num_y, np.argmin(self.dissim)%self.num_y])
        #max_fit = np.argmin(self.dissim)
        #max_fit = self.dissim.argmin()
        print 'most similar to ',input_vector,' is ',max_fit
        return max_fit


    '''
    The reweight method iteratively applies the adjust subroutine to adjust the weights of an area that falls under the
    radius (which also declines with time)

    '''
    def _reweight(self,input,center_coord,map,alpha):
        radius = int(self.radius*alpha)
        for i in range(map.shape[0]):
            for j in range(map.shape[0]):
                if np.linalg.norm(np.array([i,j])-np.array(center_coord)) > radius:
                    continue
                self._adjust(map[i][j],input,alpha)
        print 'in _reweight function'
        return

    '''
    The adjust method will adjust the weight of the given neuron (node in the map) to be closer to a target vector
    A simple application of the delta rule is applied  here
    '''

    def _adjust(self,neuron,input,alpha):
        for i in range(neuron.shape[0]):
            neuron[i]= neuron[i]+(input[i]-neuron[i])*alpha
        return

    def print_nw(self):
        for i in range(0,self.num_x):
            for j in range(0,self.num_y):
                print 'number',i,j,' : ',self.neurons[i][j],
            print ' '
        return

    def print_dissim(self):
        for i in range(0,self.num_x):
            for j in range(0,self.num_y):
                print 'co-ordinate ',i,j,' dissimilarity ',self.dissim[i][j]
        return

    '''
    Applies the _train_map method iteratively to a batch of training vectors.
    '''

    def _train_map(self,data,alpha,alpha_min,decay):
        iterations = 0
        while alpha > alpha_min:
            for d in data:
                BMU = self._get_min_dissimilarity(d)
                self._reweight(d,BMU,self.neurons,alpha)
            alpha = decay*alpha
            iterations+=1
        print 'Number of iterations = ',iterations
        return

    '''
    Applies the train_map method to one training vector
    '''

    def _train_map_one_vector(self,input_vector,alpha):

        self._reweight(input_vector,self._get_min_dissimilarity(input_vector),self.neurons,alpha)
        return


def som(data,x_size,y_size,input_dims,decay,alpha,alpha_min):

    som_map = SOM(x_size,y_size,input_dims)
    som_map.print_nw()
    som_map._train_map(data,alpha,alpha_min,decay)
    #som_map._train_map(data,alpha)
    #som_map.print_dissim()
    #som_map.print_nw()

data = ([1,0,0])
data= np.array(data)

som(data,6,12,3,0.99,0.9,0.1)
