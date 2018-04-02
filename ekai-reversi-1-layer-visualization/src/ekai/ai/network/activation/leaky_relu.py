'''
Created on 2 Apr 2018

@author: Eldemin
'''

import numpy as np

class LeakyRelu(object):

    def forward_prop(self, prop):
        return np.maximum(prop, prop * 0.01)
    
    
    def backward_prop(self, actv, d_actv):
        return np.multiply(d_actv, np.sign(actv) * 0.495 + 0.505)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    