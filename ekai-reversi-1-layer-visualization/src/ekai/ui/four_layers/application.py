'''
Created on 2 Apr 2018

@author: Eldemin
'''
from _functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.stacklayout import StackLayout

from ekai.ai.network.activation.tanh import TanH
from ekai.ai.network.dense_adam_layer import DenseAdamLayer
from ekai.ai.network.input_layer import InputLayer
from ekai.ai.reversi.player import Player
from ekai.ai.reversi.trainer import Trainer
from ekai.ui.player_ui import PlayerUI
from ekai.ui.reversi_ui import ReversiUI
from ekai.ai.network.network import Network
from ekai.ai.network.activation.leaky_relu import LeakyRelu


class Application(App):

    def create_button(self, text, callback):
        btn = Button(text=text, size_hint_x=None, size_hint_y=None, width=160, height=30)
        btn.bind(on_press=lambda instance: callback())
        self.btn_stack.add_widget(btn)

    
    def build(self):
        self.ewa_eval = 0
        self.ewa_eval_beta = 0.98
        self.eval_every = 100
        self.eval_side = 1
        
        
        
        self.clockEvent = None
        
        self.reversi_ui = ReversiUI()
        self.btn_stack = StackLayout(size_hint=[None, None], width=160)



        # build player
        learning_rate = 0.1
        
        input_layer = InputLayer(128)
        last_layer = DenseAdamLayer(input_layer, 128, learning_rate, LeakyRelu())
        for i in range(0, 10):
            last_layer = DenseAdamLayer(last_layer, 128, learning_rate, LeakyRelu())
        last_layer = DenseAdamLayer(last_layer, 1, learning_rate, TanH())
        #last_layer = DenseAdamLayer(input_layer, 1, learning_rate, TanH())
        
        self.player = Player(Network(input_layer, last_layer))
        
        self.trainer = Trainer(self.player)
        
        
        # build layout
        self.layout = StackLayout();
        self.layout.orientation = 'lr-tb'
        self.layout.add_widget(self.reversi_ui.node)
        
        # build btn stack
        self.btn_stack.orientation = 'tb-lr'
        self.btn_stack.size_hint = None, 1
        self.layout.add_widget(self.btn_stack)
        self.create_button('1 random move', self.make_random_move)
        self.create_button('1 best move', self.make_best_move)
        self.create_button('1 training move', self.make_training_move)
        self.create_button('1 training game', self.play_one_training_game)
        self.create_button('finish and train', self.finish_and_train)
        self.create_button('reset game', self.reset_game)
        self.create_button('train 1 game', self.reset_and_train_one_game)
        self.create_button('train 10 game', partial(self.train_n_games, 10))
        self.create_button('auto train', self.toggle_auto_train)
        self.create_button('eval 1 game', self.evaluate_one_game)
        self.create_button('eval/train 100 games', self.evaluate_100_games)
    
        # build player's AttrUI
        self.player_ui = PlayerUI('pickle/player', self.player_load_callback)
        self.layout.add_widget(self.player_ui.node)
        self.player_ui.set_obj(self.player)
        
        # build opponent's AttrUI
        self.opponent_ui = PlayerUI('pickle/opponent', self.opponent_load_callback)
        self.layout.add_widget(self.opponent_ui.node)
        
        # prepare board UI
        self.reversi_ui.update(self.trainer.reversi)
        self.reversi_ui.set_on_tile_clicked(self.on_tile_clicked)
        
        return self.layout

    
    def on_tile_clicked(self, pos):
        self.trainer.make_move(pos)
        self.reversi_ui.update(self.trainer.reversi)
    
    
    def player_load_callback(self, player):
        self.trainer.player = player
        self.player = player
    
    
    def opponent_load_callback(self, opponent):
        self.trainer.opponent = opponent
        self.opponent = opponent
    
    
    def make_random_move(self):
        self.trainer.make_random_move()
        self.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def make_best_move(self):
        self.trainer.make_best_move()
        self.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def make_training_move(self):
        self.trainer.make_training_move()
        self.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def play_one_training_game(self):
        self.trainer.play_one_training_game()
        self.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def finish_and_train(self):
        self.trainer.finish_and_train()
        self.player_ui.update()
    
    
    def reset_game(self):
        self.trainer.reversi.reset()
        self.reversi_ui.update(self.trainer.reversi)
    
    
    def reset_and_train_one_game(self):
        self.trainer.reversi.reset()
        self.trainer.play_one_training_game()
        self.trainer.finish_and_train()
        self.reversi_ui.update(self.trainer.reversi)
        self.player_ui.update()
        
        
    def auto_train(self, t=None):
        self.reset_and_train_one_game()
        
        if self.player.total_games % self.eval_every == 0:
            result = self.trainer.evaluate_one_game(self.eval_side)
            self.eval_side *= -1
            self.ewa_eval = self.ewa_eval_beta * self.ewa_eval + (1 - self.ewa_eval_beta) * result
            print('Evaluation! Exp: ', self.player.total_games, 'result: ', result, 'ewa: ', self.ewa_eval)
            
    
    def train_n_games(self, n):
        for i in range(0, n):
            self.trainer.reversi.reset()
            self.trainer.play_one_training_game()
            self.trainer.finish_and_train()
        
        self.reversi_ui.update(self.trainer.reversi)
        self.player_ui.update()
        
    
    def toggle_auto_train(self):
        if (self.clockEvent == None):
            self.clockEvent = Clock.schedule_interval(self.auto_train, 0)
        else:
            Clock.unschedule(self.clockEvent)
            self.clockEvent = None
        
    
    def evaluate_one_game(self):
        # TODO: random side
        self.trainer.evaluate_one_game(1)
        self.reversi_ui.update(self.trainer.reversi)
        
    
    def evaluate_100_games(self):
        # hyper settings
        training_per_evaluation = 0
        
        # result count
        wins = 0
        draws = 0
        losses = 0
        
        for i in range(0, 100):
            # TODO: random side
            result = self.trainer.evaluate_one_game(1)
            print('result: ', result)
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
            
            for j in range(0, training_per_evaluation):
                self.reset_and_train_one_game()
    
        print('total', 'wins:', wins, 'draws:', draws, 'losses:', losses)
    
    