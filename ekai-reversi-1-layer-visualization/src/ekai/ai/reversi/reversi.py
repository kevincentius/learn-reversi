
import numpy as np

class Reversi(object):
    '''
    self.board is a numpy array of dimension 2x8x8
    0 = empty, 1 = black, -1 = white
    Black plays first.
    Black starts with bottom left and top right of the center.
    '''
    
    
    def __init__(self):
        self.height = 8
        self.width = 8
        self.size = 64
        
        self.side = 1
        self.moves = 0
        self.passes = 0
        self.reset()
        
        
    def get_legal_moves(self, side=None):
        if side == None:
            side = self.side
        
        legal_moves = []
        for y in range(0, self.height):
            for x in range(0, self.width):
                if self.board[y][x] == 0:
                    for direction in self.get_check_dirs(side, y, x):
                        if not self.get_sandwiched(side, direction, y, x) == 0:
                            legal_moves.append([y, x])
                            break
                    
        return legal_moves
        
    
    def get_check_dirs(self, side, y, x):
        # find directions which contains enemy stone
        check_dirs = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ty = y+dy
                tx = x+dx
                if self.is_inside(ty, tx) and self.board[ty][tx] == -side:
                    check_dirs.append([dy, dx])
        
        return check_dirs

    def play_move(self, move):
        y = move[0]
        x = move[1]
        self.board[y][x] = self.side
        
        # flip tiles that should be flipped
        for direction in self.get_check_dirs(self.side, y, x):
            for i in range(1, self.get_sandwiched(self.side, direction, y, x) + 1):
                self.board[y + i*direction[0]][x + i*direction[1]] = self.side
        
        self.side = -self.side
        
        self.moves += 1
        self.passes = 0
        
        
    def pass_move(self):
        self.side = -self.side
        self.passes += 1
    
        
    def get_sandwiched(self, side, direction, y, x):
        for dist in range(1, max(self.width, self.height)):
            ty = y + dist * direction[0]
            tx = x + dist * direction[1]
            
            if (not self.is_inside(ty, tx)) or self.board[ty][tx] == 0:
                # not surrounded --> nothing flipped
                return 0
            elif self.board[ty][tx] == side:
                # surrounded --> flip all tiles along the direction up to distance dist
                return dist-1
        
        return 0
        
    def is_inside(self, y, x):
        return (y >= 0
            and x >= 0
            and y < self.height
            and x < self.width)
    
        
    def print(self):
        print(self.board)
        
        
    def get_input(self, move):
        y = move[0]
        x = move[1]
        
        # find tiles that should be flipped
        flipped_tiles = []
        for direction in self.get_check_dirs(self.side, y, x):
            for i in range(1, self.get_sandwiched(self.side, direction, y, x) + 1):
                flipped_tiles.append([y + i*direction[0], x + i*direction[1]])
        
        # build input
        own_mat = (self.board == self.side).astype(int)
        opponent_mat = (self.board == -self.side).astype(int)
        
        own_mat[y][x] = 1
        for flipped_tile in flipped_tiles:
            own_mat[flipped_tile[0]][flipped_tile[1]] = 1
            opponent_mat[flipped_tile[0]][flipped_tile[1]] = 0
        
        return [own_mat, opponent_mat]
        
        
    def get_winner(self):
        return np.sign((self.board == 1).sum() - (self.board == -1).sum())
        
        
    def reset(self):
        self.board = np.zeros([8, 8])
        self.board[3][4] = 1
        self.board[4][3] = 1
        self.board[3][3] = -1
        self.board[4][4] = -1
        self.side = 1
        self.moves = 0
        self.passes = 0
        
        
        
        
        