'''
Created on 2 Apr 2018

@author: Eldemin
'''

import numpy as np

class Sigmoid(object):

    def forward_prop(self, prop):
        return 1 / (1 + np.exp(-prop))
    
    
    def backward_prop(self, actv, d_actv):
        return np.multiply(d_actv, np.multiply(actv, (1 - actv)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    