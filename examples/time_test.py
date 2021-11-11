import numpy as np
import chess
import time





encoding={'.': np.array([0., 0., 0.]),
 'p': np.array([0., 0., 1.]),
 'P': np.array([ 0.,  0., -1.]),
 'b': np.array([0., 1., 0.]),
 'B': np.array([ 0., -1.,  0.]),
 'n': np.array([1., 0., 0.]),
 'N': np.array([-1.,  0.,  0.]),
 'r': np.array([0., 1., 1.]),
 'R': np.array([ 0., -1., -1.]),
 'q': np.array([1., 0., 1.]),
 'Q': np.array([-1.,  0., -1.]),
 'k': np.array([1., 1., 0.]),
 'K': np.array([-1., -1.,  0.])}


def encode(board,encoding):
    b=str(board).replace(' ','').split('\n')
    a=np.zeros([8,8,len(encoding['.'])])
    for i,row in enumerate(b):
        for j,val in enumerate(row):
            a[i,j,:]=encoding[val]
    return a


def timer(function):
    
    i = 1000
    elapsed = 0
    for _ in range(i):
        start = time.time()
        function()
        elapsed += time.time()-start
    elapsed/=i
    return elapsed

def prueba():
    board = chess.Board()
    moves=list(board.legal_moves)    
    t_moves=np.zeros([len(moves),8,8,3],dtype=np.float32)
    for i,m in enumerate(moves):
        board.push(m)
        t_moves[i,:]=encode(board,encoding)
        board.pop()
        


print("{:.4f}µs".format(timer(prueba)*1e6))