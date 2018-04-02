
import numpy as np

class DenseLayer(object):
    
    l2regularization = (1-2*0.000)

    next_layer = None
    
    # n: number of neurons in this layer
    def __init__(self, prev_layer, n, alpha, activation):
        self.n = n
        self.prev_layer = prev_layer
        self.w = np.random.rand(n, prev_layer.get_output_size()) * 0.01
        self.b = (np.random.rand(n) * 0.01).reshape(n, 1)
        self.learn_rate = alpha
        self.activation = activation
        
        prev_layer.bind(self)
    
    
    def bind(self, next_layer):
        self.next_layer = next_layer
    
    
    def get_output_size(self):
        return self.n
    
    
    # input: 2d matrix of dimension (inp_size x m)
    #    m is the number of training set
    #    each column in the input represents a training set
    # 
    # for now, leaky ReLU is used
    def forward_prop(self, inp):
        self.inp = inp;
        prop = np.dot(self.w, inp) + self.b
        self.actv = self.activation.forward_prop(prop)
        
        if self.next_layer == None:
            return self.actv
        else:
            return self.next_layer.forward_prop(self.actv)
    
    
    def backward_prop(self, d_actv):
        m = self.actv.shape[1]
        d_prop = self.activation.backward_prop(self.actv, d_actv)
        
        d_w = np.dot(d_prop, self.inp.T) / m
        d_b = np.sum(d_prop, axis=1, keepdims=True) / m
        
        prev_d_actv = np.dot(self.w.T, d_prop)
        
        # update weights
        self.w = self.w * self.l2regularization - self.learn_rate * d_w
        self.b = self.b - self.learn_rate * d_b
        
        self.prev_layer.backward_prop(prev_d_actv);
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    