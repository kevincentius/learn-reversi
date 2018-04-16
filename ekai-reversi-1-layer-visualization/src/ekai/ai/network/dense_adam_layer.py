
import numpy as np

class DenseAdamLayer(object):

    # n: number of neurons in this layer
    def __init__(self, prev_layer, n, alpha, activation):
        # hyperparameters
        self.l2regularization = (1-2*0.000) # mean squared weight loss function (weight decay)
        self.beta1 = 0.9 # momentum
        self.beta2 = 0.999 # RMSprop
        self.eps = 0.00000001
    
        # binding
        self.next_layer = None
        
        # momentum: v = exponentially weighted average
        # RMSprop: s = exponentially weighted squared average
        self.vdw = 0
        self.sdw = 0
        self.vdb = 0
        self.sdb = 0
        
        
        # build layer
        self.n = n
        self.prev_layer = prev_layer
        self.w = np.random.randn(n, prev_layer.get_output_size()) * np.sqrt(2 / prev_layer.get_output_size())
        self.b = (np.random.rand(n) * 0.0).reshape(n, 1)
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
        
        # update exponentially weighted averages
        self.vdw = self.beta1 * self.vdw + (1 - self.beta1) * d_w
        self.vdb = self.beta1 * self.vdb + (1 - self.beta1) * d_b
        self.sdw = self.beta2 * self.sdw + (1 - self.beta2) * np.multiply(d_w, d_w)
        self.sdb = self.beta2 * self.sdb + (1 - self.beta2) * np.multiply(d_b, d_b)
        
        # update weights
        self.w = self.w * self.l2regularization - self.learn_rate * self.vdw / (np.sqrt(self.sdw) + self.eps)
        self.b = self.b - self.learn_rate * self.vdb / (np.sqrt(self.sdb) + self.eps)
        
        self.prev_layer.backward_prop(prev_d_actv);
        
    
    # assuming every layer has the same learning rate
    # sets the learning rate of this and all next layers
    def set_learning_rate_all(self, alpha):
        self.learn_rate = alpha
        if self.next_layer != None:
            self.next_layer.set_learning_rate_all(alpha)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    