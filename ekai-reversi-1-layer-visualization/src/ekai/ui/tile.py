
from kivy.uix.button import Button

class Tile(Button):

    last_w = None

    def __init__(self):
        Button.__init__(self)
        self.background_normal = ''
        self.background_color = (0.1, 0.6, 0.1, 1)
    
    def update(self, team: int):
        if team == 0:
            self.background_color = (0.1, 0.6, 0.1, 1)
            self.color = (1, 1, 1, 1)
        elif team == -1:
            self.background_color = (1, 1, 1, 1)
            self.color = (0, 0, 0, 1)
        elif team == 1:
            self.background_color = (0.1, 0.1, 0.1, 1)
            self.color = (1, 1, 1, 1)
        else:
            raise Exception('Invalid team ' + team)

    def show_weight(self, w):
        if self.last_w == None or self.last_w == w:
            sign = '*'
        elif self.last_w < w:
            sign = '+'
        else:
            sign = '-'
        
        self.last_w = w
        self.text = str(round(w, 3)) + sign
        
        
    
    
    
    