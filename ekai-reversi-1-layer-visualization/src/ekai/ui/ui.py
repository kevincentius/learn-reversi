
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout

from ekai.ai.reversi.reversi import Reversi
from ekai.ui.reversi_ui import ReversiUI
from kivy.uix.widget import Widget
from kivy.uix.stacklayout import StackLayout


class UI(StackLayout):

    reversi_ui = ReversiUI()
    btn_stack = StackLayout()

    def create_button(self, text, callback):
        btn = Button(text=text, size_hint_x=None, size_hint_y=None, width=160, height=30)
        btn.bind(on_press=lambda instance: callback())
        self.btn_stack.add_widget(btn)

    def __init__(self, controller):
        super(UI, self).__init__()
        self.add_widget(self.reversi_ui.node)
        self.reversi_ui.node.pos = 50, self.reversi_ui.node.size[1]
        
        self.add_widget(self.btn_stack)
        self.create_button('1 training move', controller.make_training_move)
        self.create_button('1 training game', controller.play_one_training_game)
        self.create_button('finish and train', controller.finish_and_train)
        self.create_button('reset game', controller.reset_game)
    