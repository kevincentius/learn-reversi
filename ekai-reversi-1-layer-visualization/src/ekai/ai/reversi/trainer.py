'''
Created on 2 Apr 2018

@author: Eldemin
'''
import numpy as np
from ekai.ai.reversi.reversi import Reversi
from ekai.ai.reversi.rand_player import RandPlayer

class Trainer(object):
    
    def __init__(self, player):
        self.sum_score = 0
        
        #eval_disp
        self.eval_games = 0
        self.eval_wins = 0
        self.eval_target = 100
        
        
        
        self.player = player
        self.reversi = Reversi()
        self.inp_array = []
        self.side_array = []
        self.opponent = RandPlayer()
        
    
    def play_one_training_game(self):
        while self.reversi.moves < 60 and self.reversi.passes < 2:
            self.make_training_move()
        
    
    def make_random_move(self):
        legal_moves = self.reversi.get_legal_moves()
        if len(legal_moves) > 0:
            move = self.player.pick_random_move(legal_moves)
            self.reversi.play_move(move)
        else:
            self.reversi.pass_move()


    def make_best_move(self):
        legal_moves = self.reversi.get_legal_moves()
        if len(legal_moves) > 0:
            move = self.player.pick_best_move(self.reversi, legal_moves)
            self.reversi.play_move(move)
        else:
            self.reversi.pass_move()

    
    def make_training_move(self):
        legal_moves = self.reversi.get_legal_moves()
        
        if len(legal_moves) == 0:
            self.reversi.pass_move()
        else:
            move = self.player.pick_training_move(self.reversi, legal_moves)
            self.append_move_to_history(move)
            self.reversi.play_move(move)

    
    def make_move(self, move):
        if move in self.reversi.get_legal_moves():
            self.reversi.play_move(move)
        else:
            print('illegal move ', move)
    
    

    def make_opponent_move(self):
        legal_moves = self.reversi.get_legal_moves()
        if len(legal_moves) > 0:
            move = self.opponent.pick_best_move(self.reversi, legal_moves)
            self.reversi.play_move(move)
        else:
            self.reversi.pass_move()


    def reshape_input(self, cinp):
        return np.row_stack([cinp[0].reshape([64, 1]), cinp[1].reshape([64, 1])])


    def append_move_to_history(self, move):
        # save history
        cinp = self.reversi.get_input(move)
        
        for rot in range(0, 4):
            # append original position
            self.inp_array.append(self.reshape_input(cinp))
            self.side_array.append(self.reversi.side)
            
            # append transposed position (data augmentation)
            self.inp_array.append(self.reshape_input([cinp[0].T, cinp[1].T]))
            self.side_array.append(self.reversi.side)
            
            # rotate position (data augmentation)
            cinp[0] = np.rot90(cinp[0])
            cinp[1] = np.rot90(cinp[1])
        

    def finish_and_train(self):
        # prepare input
        inp = np.column_stack(self.inp_array)
        
        # prepare output
        # if white wins, flip the side
        # if draw, treat as if black wins, i.e. do not flip the side
        out = np.asarray(self.side_array).reshape(1, -1)
        if self.reversi.get_winner() == -1:
            out = np.multiply(out, -1)
        
        # clear data from pipeline
        self.inp_array = []
        self.side_array = []
        
        # train network
        self.player.train_game(inp, out)
        
        # evaluation
        self.sum_score += abs(self.reversi.board.sum())


    # TODO: some randomness so we can evaluate vs opponent
    def evaluate_one_game(self, side):
        self.reversi.reset()
        
        while self.reversi.moves < 60 and self.reversi.passes < 2:
            if self.reversi.side == side:
                self.make_best_move()
            else:
                self.make_opponent_move()
        
        #print('side: ', side, 'winner: ', self.reversi.get_winner(), 'result: ', side * self.reversi.get_winner())
        return side * self.reversi.get_winner()
        
        
    def start_eval(self, num_games):
        self.eval_target = 100
        self.eval_games = 0
        self.eval_wins = 0
        