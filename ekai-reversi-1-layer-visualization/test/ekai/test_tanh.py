'''
Created on 2 Apr 2018

@author: Eldemin
'''
import unittest

from ekai.ai.network.activation.tanh import TanH
from ekai.ai.network.dense_layer import DenseLayer
from ekai.ai.network.input_layer import InputLayer
from ekai.ai.network.network import Network
import numpy as np


class Test(unittest.TestCase):

    def testName(self):
        inp_size = 20
        
        input_layer = InputLayer(inp_size)
        dense_layer = DenseLayer(input_layer, 1, 0.025, TanH())
        network = Network(input_layer, dense_layer)
        
        # training
        for i in range(0, 100000):
            inp = np.random.rand(inp_size, 1)
            if np.sum(inp, axis=0, keepdims=True) > inp_size / 2:
                out = 1
            else:
                out = -1
            
            network.train(inp, out)
            
        # evaluation
        for i in range(0, 100):
            inp = np.random.rand(inp_size, 1)
            nout = network.forward_prop(inp)
            print(np.sum(inp, axis=0, keepdims=True), nout)
            
        
        print(dense_layer.w)
        print('b:', dense_layer.b)
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()