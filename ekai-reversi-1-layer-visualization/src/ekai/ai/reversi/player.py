'''
Created on 1 Apr 2018

@author: Eldemin
'''
from math import inf

import numpy as np


class Player(object):

    def __init__(self, network):
        self.name = 'untitled'
    
        self.learning_rate = 0.0015
        self.exploration_min = 0.04
        self.exploration_decay = 0.999
        
        self.exploration = 1
        self.total_games = 0
        
        
    
        self.network = network
        self.network.set_learning_rate(self.learning_rate)
        
    
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

    
    def train_game(self, inp, out):
        self.network.train(inp, out)
        # decrease exploration
        self.exploration = max(self.exploration_min, self.exploration * self.exploration_decay)
        # stats
        self.total_games += 1
        
        
    def on_update_learning_rate(self):
        self.network.set_learning_rate(self.learning_rate)
        
        