
from kivy.uix.gridlayout import GridLayout
from ekai.ui.tile import Tile
from ekai.ai.reversi.reversi import Reversi

class ReversiUI(object):

    def __init__(self):
        self.tile_size = 60
        self.spacing = 2
        
        self.node = GridLayout(cols = 8, rows = 8, size_hint_x = None, size_hint_y = None)
        self.tiles = []
        
        self.on_tile_clicked = None



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
                self.__bind_tile_click(tile, y, x)
            self.tiles.append(row)
        
        self.node.width = self.tile_size * 8 + self.spacing * 7
        self.node.height = self.tile_size * 8 + self.spacing * 7
            
            
    def __bind_tile_click(self, tile, y, x):
        tile.bind(on_press=lambda instance: self.__callback([y, x]))
        
    
    def __callback(self, pos):
        if self.on_tile_clicked is not None:
            self.on_tile_clicked(pos)
    
    
    def update(self, reversi: Reversi):
        for y in range(0, 8):
            for x in range(0, 8):
                self.tiles[y][x].update(reversi.board[y][x])
    
    
    def set_on_tile_clicked(self, on_tile_clicked):
        self.on_tile_clicked = on_tile_clicked
    