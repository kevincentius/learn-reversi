'''
Created on 1 Apr 2018

@author: Eldemin
'''
from math import inf

from ekai.ai.network.dense_layer import DenseLayer
from ekai.ai.network.input_layer import InputLayer
from ekai.ai.network.network import Network
import numpy as np
from ekai.ai.network.activation.tanh import TanH


class Player(object):

    learning_rate = 0.01
    exploration = 1
    total_games = 0
    
    def __init__(self):
        self.input_layer = InputLayer(128)
        self.dense_layer = DenseLayer(self.input_layer, 1, self.learning_rate, TanH())
        self.network = Network(self.input_layer, self.dense_layer)
        
    
    def reshape_input(self, cinp):
        return np.row_stack([cinp[0].reshape([64, 1]), cinp[1].reshape([64, 1])])


    def pick_best_move(self, reversi, legal_moves):
        best_score = float(-inf)
        for move in legal_moves:
            score = self.network.forward_prop(self.reshape_input(reversi.get_input(move)))
            if score > best_score:
                best_move = move
                best_score = score
        
        return best_move


    def pick_random_move(self, legal_moves):
        return legal_moves[np.random.randint(len(legal_moves))]


    def pick_training_move(self, reversi, legal_moves):
        move = None
    # exploration is chance to explore random move instead of picking the best move
        if (np.random.rand() > self.exploration):
            move = self.pick_best_move(reversi, legal_moves)
        else:
            move = self.pick_random_move(legal_moves)
        return move

