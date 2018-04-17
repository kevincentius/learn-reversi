
import numpy as np
from ekai.ai.network.loss.SquaredError import SquaredError

class Network(object):

    def __init__(self, input_layer, output_layer, loss_func=SquaredError()):
        
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.loss_func = loss_func
        
    
    def train(self, inp, out):
        a = self.forward_prop(inp)
        
        # evaluate
        #print('w:', self.input_layer.next_layer.w)
        print('a:', a)
        #print('cost: ', np.sum(self.loss_func.calc_loss(a, out)) / out.shape[1], 'shape: ', out.shape[1])
        
        # backward propagation
        d_actv = self.loss_func.calc_d_actv(a, out)
        self.output_layer.backward_prop(d_actv)
    
    
    def forward_prop(self, inp):
        return self.input_layer.forward_prop(inp)
    
    
    # assuming every layer has the same learning rate
    def set_learning_rate(self, alpha):
        self.input_layer.next_layer.set_learning_rate_all(alpha)
        
        
        
        