
class OutputLayer(object):

    def __init__(self, prev_layer):
        self.prev_layer = prev_layer
        prev_layer.bind(self)
    
    
    def get_output_size(self):
        return self.prev_layer.get_output_size()
    
    
    def set_expected_output(self, y):
        self.y = y
            
    
    def forward_prop(self, inp):
        return inp
    
    
    def backward_prop(self, dactv):
        self.prev_layer.backward_prop(dactv)