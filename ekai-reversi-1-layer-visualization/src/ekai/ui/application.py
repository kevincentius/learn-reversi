'''
Created on 2 Apr 2018

@author: Eldemin
'''
from ekai.ui.ui import UI
from kivy.app import App
from ekai.ai.reversi.player import Player
from ekai.ai.reversi.trainer import Trainer

class Application(App):

    player = Player()
    trainer = Trainer(player)
    
    def build(self):
        self.ui = UI(self)
        self.ui.reversi_ui.update(self.trainer.reversi)
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        return self.ui

    
    def make_training_move(self):
        self.trainer.make_training_move()
        self.ui.reversi_ui.update(self.trainer.reversi)
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        pass
    
    
    def play_one_training_game(self):
        self.trainer.play_one_training_game()
        self.ui.reversi_ui.update(self.trainer.reversi)
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
        pass
    
    
    def finish_and_train(self):
        self.trainer.finish_and_train()
        self.ui.reversi_ui.show_weights(self.player.dense_layer.w)
    
    
    def reset_game(self):
        self.trainer.reversi.reset()
        self.ui.reversi_ui.update(self.trainer.reversi)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    