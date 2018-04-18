import pickle

from numpy import float64

from ekai.ai.network.activation.leaky_relu import LeakyRelu
from ekai.ai.network.activation.sigmoid import Sigmoid
from ekai.ai.network.activation.tanh import TanH
from ekai.ai.network.dense_adam_layer import DenseAdamLayer
from ekai.ai.network.input_layer import InputLayer
from ekai.ai.network.loss.CrossEntropy import CrossEntropy
from ekai.ai.network.loss.SquaredError import SquaredError
from ekai.ai.network.network import Network
from ekai.ai.network.output_layer import OutputLayer
from ekai.ui.four_layers.application import Application
import numpy as np
from ekai.ai.network.dense_layer import DenseLayer


#kivy.require('1.0.6') # replace with your current kivy version !
def test_load_and_evaluate():
    network = pickle.load(open('pickle/network.pickle', 'rb'))
    for i in range(0, 100):
        inp = np.random.rand(363, 1)
        out = np.sum(inp, keepdims=True)
        print(out, '->', network.forward_prop(inp))



def test_load_and_train_alpha(alpha):
    network = pickle.load(open('pickle/network.pickle', 'rb'))
    network.set_learning_rate(alpha)
    
    for i in range(0, 5000):
        inp = np.random.rand(363, 1)
        out = np.sum(inp, keepdims=True)
        network.train(inp, out)
    
    pickle.dump(network, open('pickle/network.pickle', "wb"))



def test_learn_from_scratch():
    #input_layer = InputLayer(363)
    #dense_layer = DenseAdamLayer(input_layer, 1, 0.001, LeakyRelu())
    #output_layer = OutputLayer(dense_layer)
    #network = Network(input_layer, output_layer)
    
    
    # build player
    learning_rate = 0.025
    
    input_layer = InputLayer(512)
    #last_layer = DenseAdamLayer(input_layer, 128, learning_rate, LeakyRelu())
    #last_layer = DenseAdamLayer(last_layer, 128, learning_rate, LeakyRel))
    last_layer = DenseAdamLayer(input_layer, 1, learning_rate, Sigmoid())
    
    network = Network(input_layer, last_layer, CrossEntropy())
    
    for i in range(0, 10000):
        inp = np.random.rand(512, 1000)
        out = np.greater(np.sum(inp, axis=0, keepdims=True), 256).astype(float64)
        #print('sum', np.sum(inp, keepdims=True), 'out', out)
        #print('a', network.forward_prop(inp))
        #print('w', network.input_layer.next_layer.w)
        #print('equal:', np.equal((np.sign(network.forward_prop(inp)-0.5)+1) / 2, out))
        print('accuracy:', np.sum(np.equal((np.sign(network.forward_prop(inp)-0.5)+1) / 2, out)))
        network.train(inp, out)
    
    #pickle.dump(network, open('pickle/network.pickle', "wb"))


if __name__ == '__main__':
    #Application().run()
    test_learn_from_scratch()
    #test_load_and_evaluate()
    #test_load_and_train_alpha(0.00001)
    #player.train(reversi, 25)