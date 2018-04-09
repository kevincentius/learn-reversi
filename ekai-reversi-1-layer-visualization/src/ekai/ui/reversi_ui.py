
from kivy.uix.gridlayout import GridLayout
from ekai.ui.tile import Tile
from ekai.ai.reversi.reversi import Reversi

class ReversiUI(object):
    
    tile_size = 60
    spacing = 2
    
    node = GridLayout(cols = 8, rows = 8, size_hint_x = None, size_hint_y = None)
    tiles = []

    def __init__(self):
        self.node.row_force_default = True
        self.node.col_force_default = True
        self.node.row_default_height = self.tile_size
        self.node.col_default_width = self.tile_size
        self.node.spacing = self.spacing
        
        for y in range(0, 8):
            row = []
            for x in range(0, 8):
                tile = Tile()
                row.append(tile)
                self.node.add_widget(tile)
            self.tiles.append(row)
        
        self.node.width = self.tile_size * 8 + self.spacing * 7
        self.node.height = self.tile_size * 8 + self.spacing * 7
            
    
    def update(self, reversi: Reversi):
        for y in range(0, 8):
            for x in range(0, 8):
                self.tiles[y][x].update(reversi.board[y][x])
    
    
    
    
    