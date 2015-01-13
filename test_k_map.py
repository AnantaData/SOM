import unittest
from Refactored_SOM import SOM
import numpy as np
import copy
import random

__author__ = 'laksheen'

class TestK_map(unittest.TestCase):

    def setUp(self):

        print 'in setup'
        self.map = SOM(5,5,16)
        self.map.print_nw()

        data = np.loadtxt("zoo.data.txt",dtype=str,delimiter=",")
        data = np.array(data)
        names = data[:,0]
        names= np.column_stack((names,data[:,-1]))
        self.features= data[:,:-1]
        #print features
        self.features = self.features[:,1:].astype(int)
        #map= SOM(5,5,16)

    def tearDown(self):

        print 'in tearDown'
        del self.features
        del self.map

    def test_get_min_dissimilarity_not_none(self):

        print 'in test1'
        #map.neurons = np.ones(shape=(5,5,16))
        print self.features[0]
        self.assertIsNotNone(self.map._get_min_dissimilarity(self.features[0]),"returned nothing")

    def test_get_min_dissimilarity_correct_neuron_returned(self):

        print 'in test2'
        self.map.neurons[3][0]=self.features[12]                                    #map.neurons[3][0] is same as features[12], thus min dissim is neuron [3][0]
        print self.map.neurons[3][0]
        self.map.print_nw()
        self.assertTrue(np.alltrue(self.map._get_min_dissimilarity(self.features[12]) == np.array([3 ,0])))

    def test_get_min_dissimilarity_best_neuron_returned(self):

        print 'in test3'
        self.map.neurons[1][1]= np.array([1,0,0,1,0,0,0,1,1,1,0,1,4,1,0,1])         #map.neurons[1][1] is slightly different from features[28], thus min dissim is neuron [1][1]
        self.map.print_nw()
        self.assertTrue(np.alltrue(self.map._get_min_dissimilarity(self.features[28]) == np.array([1, 1])))

    def test_adjust(self):

        print 'in test4'
        original_weight = copy.copy(self.map.neurons[3][2])
        self.map._adjust(self.map.neurons[3][2],self.features[3],alpha=0.9)

        self.assertFalse(np.alltrue( self.map.neurons[3][2] != original_weight))
        self.map.print_nw()

    def test_reweight(self):

        print 'in test 5'
        alpha =0.9
        radius = int(self.map.radius*alpha)
        original_values = copy.copy(self.map.neurons)
        random.seed()
        center_cood_x= random.randint(0,4)
        center_cood_y= random.randint(0,4)

        center_cood = np.array([center_cood_x,center_cood_y])

        print 'center coordinate is ',center_cood
        self.map._reweight(self.features[7],center_cood,self.map.neurons,alpha)

        self.map.print_nw()

        list = []

        for i in range(original_values.shape[0]):
            for j in range(original_values.shape[0]):
                if np.all(original_values[i][j] == self.map.neurons[i][j]) :
                    list.append(True)
                    continue
                list.append(False)

        print list

        self.assertFalse(np.all(list))

    def test_reweight_within_radius(self):

        print 'in test 6'

        alpha =0.9
        radius = int(self.map.radius*alpha)
        print 'radius is ',radius
        original_values = copy.copy(self.map.neurons)
        random.seed()
        center_cood_x= random.randint(0,4)
        center_cood_y= random.randint(0,4)

        center_cood = np.array([center_cood_x,center_cood_y])

        print 'center coordinate is ',center_cood
        self.map._reweight(self.features[7],center_cood,self.map.neurons,alpha)

        self.map.print_nw()

        list = []

        for i in range(original_values.shape[0]):
            for j in range(original_values.shape[0]):
                if np.all(original_values[i][j] == self.map.neurons[i][j]):
                    continue
                elif np.linalg.norm(np.array([i,j])-np.array(center_cood)) <= radius:           #check whether the neurons which have changed weights are within the radius of the center coordinate
                    print 'dissimilarity between center coordinate and neuron',i,j,' is ',radius
                    #word = np.array([i,j])
                    list.append(True)
                else:
                    list.append(False)

        print list

        self.assertTrue(np.all(list))


if __name__ == '__main__':
    unittest.main()