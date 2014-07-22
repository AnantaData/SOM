# coding: utf-8
import random
import numpy as np
import math
import sys


class k_map(object):
    
    
    
    num_x=0
    num_y=0
    inps=0
    neurons=[]
    temp=None
    def __init__(self,x_size,y_size,input_dims,decay,alpha_start,alpha_min,max_clusters):
        #self.neurons=np.empty([x_size,y_size])
        self.d_rate=decay
        self.al_min=alpha_min
        self.alpha=alpha_start
        self.num_x=x_size
        self.num_y=y_size
        self.inps=input_dims
        self.m_clus=max_clusters
        self.neurons=np.random.randint(100,size=(x_size,y_size,input_dims))
        self.dissim=[[]]*max_clusters
        #self.neurons=[[neuron(input_dims)for y in range(0,y_size)]for x in range(0,x_size)]
        
    def print_nw(self):
        for i in range(0,self.num_x):
            for j in range(0,self.num_y):
                print 'number',i,j
                print self.neurons[i][j]
        return
                
        
    def insert(self,inp_vector):
        for i in range(self.m_clus):
            self.dissim[i]=0.0
        for i in range(self.m_clus):
            self.dissim[i]+=np.linalg.norm(self.neurons[i/self.num_x][i%self.num_y]-inp_vector)
        return
    
    
    def train_map(self,patterns, training_set):
        self.iters=0
        
        while self.alpha>self.al_min:
            self.iters+=1
            
            for i in range(training_set.shape[0]):
                self.insert(training_set[i])
                
                mind= min(self.dissim)
                
                clus_mind=np.where(dissim==mind)[0]
                
                for j in range(self.input_dims):
                    self.neurons[clus_mind/self.num_x][clus_mind%self.num_y][j]+=self.alpha*(patterns[i][j]-self.neurons[clus_mind/self.num_x][clus_mind%self.num_y][j])
            
            self.alpha=self.d_rate*self.alpha
        return
    
     
