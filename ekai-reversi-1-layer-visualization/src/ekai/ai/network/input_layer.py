
class InputLayer(object):

    def __init__(self, n):
        self.n = n
    
    
    def bind(self, next_layer):
        self.next_layer = next_layer
    
    
    def get_output_size(self):
        return self.n
    
    
    def forward_prop(self, inp):
        return self.next_layer.forward_prop(inp)
    
    
    def backward_prop(self, dactv):
        pass
    