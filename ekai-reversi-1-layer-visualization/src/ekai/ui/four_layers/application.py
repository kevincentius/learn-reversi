'''
Created on 2 Apr 2018

@author: Eldemin
'''
from _functools import partial

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.stacklayout import StackLayout

from ekai.ai.reversi.player import Player
from ekai.ai.reversi.trainer import Trainer
from ekai.ui.player_ui import PlayerUI
from ekai.ui.reversi_ui import ReversiUI


class Application(App):

    player = Player()
    trainer = Trainer(player)
    
    clockEvent = None
    
    reversi_ui = ReversiUI()
    btn_stack = StackLayout(size_hint=[None, None], width=160)

    def create_button(self, text, callback):
        btn = Button(text=text, size_hint_x=None, size_hint_y=None, width=160, height=30)
        btn.bind(on_press=lambda instance: callback())
        self.btn_stack.add_widget(btn)

    
    def build(self):
        self.layout = StackLayout();
        self.layout.orientation = 'lr-tb'
        self.layout.add_widget(self.reversi_ui.node)
        
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
    
        self.player_ui = PlayerUI(self.player_load_callback)
        self.layout.add_widget(self.player_ui.node)
        
        
        
        self.reversi_ui.update(self.trainer.reversi)
        self.reversi_ui.show_weights(self.player.dense_layer.w)
        self.player_ui.set_obj(self.player)
        return self.layout

    
    def player_load_callback(self, player):
        self.trainer.player = player
        self.player = player
        self.reversi_ui.show_weights(self.player.dense_layer.w)
    
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
        self.reversi_ui.show_weights(self.player.dense_layer.w)
        self.player_ui.update()
    
    
    def reset_game(self):
        self.trainer.reversi.reset()
        self.reversi_ui.update(self.trainer.reversi)
    
    
    def reset_and_train_one_game(self, t=None):
        self.trainer.reversi.reset()
        self.trainer.play_one_training_game()
        self.trainer.finish_and_train()
        self.reversi_ui.update(self.trainer.reversi)
        self.reversi_ui.show_weights(self.player.dense_layer.w)
        self.player_ui.update()
        
    
    def train_n_games(self, n):
        for i in range(0, n):
            self.trainer.reversi.reset()
            self.trainer.play_one_training_game()
            self.trainer.finish_and_train()
        
        self.reversi_ui.update(self.trainer.reversi)
        self.reversi_ui.show_weights(self.player.dense_layer.w)
        self.player_ui.update()
        
    
    def toggle_auto_train(self):
        if (self.clockEvent == None):
            self.clockEvent = Clock.schedule_interval(self.reset_and_train_one_game, 0)
        else:
            Clock.unschedule(self.clockEvent)
            self.clockEvent = None
        
    
    
    
    
    
    
    
    
    
    
    
    
    