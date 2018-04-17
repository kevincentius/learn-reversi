
import numpy as np

class SquaredError(object):
    
    def calc_loss(self, a, out):
        loss = (out - a)
        loss = np.multiply(loss, loss)
        return loss
    
    
    def calc_d_actv(self, a, out):
        return 2 * (a - out)