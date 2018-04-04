
from kivy.uix.button import Button

class Tile(Button):

    last_w_own = None
    last_w_opp = None

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


    def get_weight_text(self, last_w, w):
        if last_w == None or last_w == w:
            sign = '*'
        elif last_w < w:
            sign = '+'
        else:
            sign = '-'
        return str(round(w, 3)) + sign

    def show_weight(self, w_own, w_opp):
        self.text = self.get_weight_text(self.last_w_own, w_own)
        self.text += '\n' + self.get_weight_text(self.last_w_opp, w_opp)
        self.last_w_own = w_own
        self.last_w_opp = w_opp
        
        
    
    
    
    