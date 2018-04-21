
import numpy as np

class CrossEntropy(object):
    
    def calc_loss(self, a, out):
        return -(np.dot(out, np.log(a)) + np.dot(1-out, np.log(1-a)))
    
    
    def calc_d_actv(self, a, out):
        return -(out/a) + ((1-out)/(1-a))