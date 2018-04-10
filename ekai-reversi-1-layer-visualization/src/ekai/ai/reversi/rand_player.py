
import numpy as np

class RandPlayer(object):

    def pick_best_move(self, reversi, legal_moves):
        return legal_moves[np.random.randint(len(legal_moves))]

