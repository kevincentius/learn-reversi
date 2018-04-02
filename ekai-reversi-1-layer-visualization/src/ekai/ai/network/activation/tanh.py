'''
Created on 2 Apr 2018

@author: Eldemin
'''

import numpy as np

class TanH(object):

    def forward_prop(self, prop):
        return 2 / (1 + np.exp(-2*prop)) - 1
    
    
    def backward_prop(self, actv, d_actv):
        return np.multiply(d_actv, 1 - np.multiply(actv, actv))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    