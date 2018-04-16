'''
Created on 4 Apr 2018

@author: Eldemin
'''
from ekai.ui.attr_ui import AttrUI


class PlayerUI(AttrUI):
    
    def __init__(self, pickle_path, load_callback):
        AttrUI.__init__(self, [
            ['name', 'name', 's'],
            ['learning_rate', 'alpha'],
            ['exploration_min', 'min. explr'],
            ['exploration_decay', 'decay explr'],
            ['exploration', 'exploration'],
            ['total_games', 'experience'],
        ], pickle_path, load_callback);
    
    