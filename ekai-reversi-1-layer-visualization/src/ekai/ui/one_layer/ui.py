
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout

from ekai.ai.reversi.reversi import Reversi
from ekai.ui.reversi_ui import ReversiUI
from kivy.uix.widget import Widget
from kivy.uix.stacklayout import StackLayout
from _functools import partial
from ekai.ui.player_ui import PlayerUI
from ekai.ui.attr_ui import AttrUI


class UI(StackLayout):

    reversi_ui = ReversiUI()
    btn_stack = StackLayout(size_hint=[None, None], width=160)

    def create_button(self, text, callback):
        btn = Button(text=text, size_hint_x=None, size_hint_y=None, width=160, height=30)
        btn.bind(on_press=lambda instance: callback())
        self.btn_stack.add_widget(btn)

    def __init__(self, controller):
        super(UI, self).__init__()
        self.orientation = 'lr-tb'
        self.add_widget(self.reversi_ui.node)
        
        self.btn_stack.orientation = 'tb-lr'
        self.btn_stack.size_hint = None, 1
        self.add_widget(self.btn_stack)
        self.create_button('1 random move', controller.make_random_move)
        self.create_button('1 best move', controller.make_best_move)
        self.create_button('1 training move', controller.make_training_move)
        self.create_button('1 training game', controller.play_one_training_game)
        self.create_button('finish and train', controller.finish_and_train)
        self.create_button('reset game', controller.reset_game)
        self.create_button('train 1 game', controller.reset_and_train_one_game)
        self.create_button('train 10 game', partial(controller.train_n_games, 10))
        self.create_button('auto train', controller.toggle_auto_train)
    
        self.player_ui = PlayerUI(controller.player_load_callback)
        self.add_widget(self.player_ui.node)