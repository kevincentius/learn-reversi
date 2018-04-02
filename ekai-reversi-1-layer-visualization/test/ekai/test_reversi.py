'''
Created on 21 Mar 2018

@author: Eldemin
'''
import unittest
import numpy as np
from ekai.reversi import Reversi


class Test(unittest.TestCase):


    def test_play_move(self):
        reversi = Reversi()
        reversi.play_move(2, 3)
        reversi.play_move(2, 2)
        reversi.play_move(5, 4)
        reversi.play_move(1, 1)
        
        self.assertTrue(np.array_equal(np.array( \
                        [[ 0,  0,  0,  0,  0,  0,  0,  0.], \
                         [ 0, -1,  0,  0,  0,  0,  0,  0.], \
                         [ 0,  0, -1,  1,  0,  0,  0,  0.], \
                         [ 0,  0,  0, -1,  1,  0,  0,  0.], \
                         [ 0,  0,  0,  1,  1,  0,  0,  0.], \
                         [ 0,  0,  0,  0,  1,  0,  0,  0.], \
                         [ 0,  0,  0,  0,  0,  0,  0,  0.], \
                         [ 0,  0,  0,  0,  0,  0,  0,  0.]]), reversi.board))
        pass
    
    def test_get_input(self):
        reversi = Reversi()
        inp = reversi.get_input([2, 3])
        
        #                         side + y*8+x
        self.assertEqual(1, inp[0*64 + 2*8+3][0])
        self.assertEqual(1, inp[0*64 + 3*8+3][0])
        self.assertEqual(1, inp[0*64 + 3*8+4][0])
        self.assertEqual(1, inp[0*64 + 4*8+3][0])
        self.assertEqual(1, inp[1*64 + 4*8+4][0])
        
        self.assertEqual(5, np.sum(inp))
        print(inp)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()