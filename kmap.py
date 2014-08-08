# coding: utf-8
#Import the necessary dependencies
import random
import numpy as np
import math
import sys

class k_map(object):
    
    iters=0
    num_x=0
    num_y=0
    inps=0
    neurons=[]
    temp=None
    radius=0
    
    '''
    Constructor Function: Takes as input the following parameters
    
    x_size : the width of the map desired

    y_size : the height of the  map desired
    
    input_dims : the dimensinality of the vectors the som is trained to deal with
    
    decay: A constant decay rate, which is actually a learning rate. The coefficients of the parts where I  use the 
    'delta rule' will continually be multiplied by the decay to reduce it. Hence the 0 < decay < 1 (exclusive)
    
    alpha_start: Starting constant of the coefficient of delta. This should also be 1 >= alpha > 0. This will gradually
    decline owing to the decay rate
    
    alpha_min: the termination of batch learning will be determined when the alpha is lower than this bound. Hence this
    is the lower bound for this.
    
    '''
    
    def __init__(self,x_size,y_size,input_dims,decay,alpha_start,alpha_min):
        self.d_rate=decay
        self.al_min=alpha_min
        self.alpha=alpha_start
        self.num_x=x_size
        self.num_y=y_size
        self.inps=input_dims
        self.neurons=np.random.randint(100,size=(x_size,y_size,input_dims))
        self.dissim=np.ndarray(shape=(self.num_x,self.num_y))
        self.radius=min(x_size,y_size)/2
        
    def print_nw(self):
        for i in range(0,self.num_x):
            for j in range(0,self.num_y):
                print 'number',i,j
                print self.neurons[i][j]
        return
    
    '''
    The adjust method will adjust the weight of the given neuron (node in the map) to be closer to a target vector
    A simple application of the delta rule is applied  here
    '''
    
    def adjust(self,neuron,target,alpha):
        for i in range(neuron.shape[0]):
            neuron[i]= neuron[i]+(target[i]-neuron[i])*alpha
            
        return    
    
    '''
    The reweight method iteratively applies the adjust subroutine to adjust the weights of an area that falls under the
    radius (which also declines with time)
    
    '''
    
    def reweight(self,target,center,nmap,alpha):
        radius = int(self.radius*alpha)
        for i in range(nmap.shape[0]):
            for j in range(nmap.shape[0]):
                if np.linalg.norm(np.array([i,j])-np.array(center)) > radius:
                    continue
                self.adjust(nmap[i][j],target,alpha)            
        return
    

    
################################################################################################
    '''
    calc_dissims calculate the dissimilarity measures (here the eculidean distance) to all the neurons for a given
    training/input vector and returns the closestly similar vector's coordinates as output
    '''

    def calc_dissims(self,input_vector):
        self.dissim=np.zeros(shape=(self.num_x,self.num_y))
        for x in range(self.neurons.shape[0]):
            for y in range(self.neurons.shape[1]):
                self.dissim[x][y] = np.linalg.norm(self.neurons[x][y]-input_vector)
        max_fit=np.array([np.argmin(self.dissim)%self.num_x, np.argmin(self.dissim)/self.num_x])        
        return max_fit
    
################################################################################################
    '''
    Train a map with a single input. 
    '''
    def train_map(self,input_vector):
        
        self.reweight(input_vector,self.calc_dissims(input_vector),self.neurons,self.alpha)
        return
    '''
    Applies the train_map method iteratively to a batch of training vectors. Encouraged to use this rather than the
    single training method
    '''
    def batch_train(self,input_batch):
        print 'alpha = ',self.alpha,'alpha_min=',self.al_min
        while self.alpha > self.al_min:
            for inp in input_batch:
                self.train_map(inp)
            self.alpha = self.d_rate*self.alpha
            self.iters+=1
        print self.alpha
        print 'done in ',self.iters, 'iterations...'
        return
    
    '''
    gives as output the coordinates of the neuron of which the weights bear the closest resemblance to the input
    vector. 
    '''
    def predict(self,input_vector):
        
        output=self.calc_dissims(input_vector)
        
        return output
        
    '''
    The above, applied in a batch manner. Please try to figure out what is wrong with this thing...
    '''
    def cluster(self,input_batch):
        
        target=np.ndarray(shape=(input_batch.shape[0]))
        i=0;
        for inp in input_batch:
            target[i]=np.array(self.predict(inp))
        print 'done...'
        return target
        
 
