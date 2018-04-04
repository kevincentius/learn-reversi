'''
Created on 2 Apr 2018

@author: Eldemin
'''
from ekai.ui.one_layer.ui import UI
from kivy.app import App
from ekai.ai.reversi.player import Player
from ekai.ai.reversi.trainer import Trainer
from kivy.clock import Clock

class Application(App):

    player = Player()
    trainer = Trainer(player)
    
    clockEvent = None
    
    def build(self):
        self.ui = UI(self)
        self.ui.reversi_ui.update(self.trainer.reversi)
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        self.ui.player_ui.set_obj(self.player)
        return self.ui

    
    def player_load_callback(self, player):
        self.trainer.player = player
        self.player = player
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
    
    def make_random_move(self):
        self.trainer.make_random_move()
        self.ui.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def make_best_move(self):
        self.trainer.make_best_move()
        self.ui.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def make_training_move(self):
        self.trainer.make_training_move()
        self.ui.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def play_one_training_game(self):
        self.trainer.play_one_training_game()
        self.ui.reversi_ui.update(self.trainer.reversi)
        pass
    
    
    def finish_and_train(self):
        self.trainer.finish_and_train()
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        self.ui.player_ui.update()
    
    
    def reset_game(self):
        self.trainer.reversi.reset()
        self.ui.reversi_ui.update(self.trainer.reversi)
    
    
    def reset_and_train_one_game(self, t=None):
        self.trainer.reversi.reset()
        self.trainer.play_one_training_game()
        self.trainer.finish_and_train()
        self.ui.reversi_ui.update(self.trainer.reversi)
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        self.ui.player_ui.update()
        
    
    def train_n_games(self, n):
        for i in range(0, n):
            self.trainer.reversi.reset()
            self.trainer.play_one_training_game()
            self.trainer.finish_and_train()
        
        self.ui.reversi_ui.update(self.trainer.reversi)
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        self.ui.player_ui.update()
        
    
    def toggle_auto_train(self):
        if (self.clockEvent == None):
            self.clockEvent = Clock.schedule_interval(self.reset_and_train_one_game, 0)
        else:
            Clock.unschedule(self.clockEvent)
            self.clockEvent = None
        
    
    
    
    
    
    
    
    
    
    
    
    
    