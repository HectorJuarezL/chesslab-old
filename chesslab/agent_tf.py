import chess
import numpy as np
import tensorflow as tf
from .training_tf import load_model

def encode(board,encoding):
    b=str(board).replace(' ','').split('\n')
    a=np.zeros([8,8,len(encoding['.'])])
    for i,row in enumerate(b):
        for j,val in enumerate(row):
            a[i,j,:]=encoding[val]
    return a

class agent():

    def __init__(self,model,path_model):
        physical_devices = tf.config.list_physical_devices('GPU')
        print(physical_devices)
        self.encoding,self.history=load_model(model,path_model)
        self.model=model
        self.channels=len(self.encoding['.'])
        
    def get_move_values(self,board,both_players = False):
        moves=list(board.legal_moves)

        if len(moves)>0:
            t_moves=np.zeros([len(moves),8,8,self.channels],dtype=np.float32)
            for i,m in enumerate(moves):
                board.push(m)
                t_moves[i,:]=encode(board,self.encoding)
                board.pop()
            score=self.model(t_moves)
            score=tf.nn.softmax(score,1)
            score=score.numpy()
            if not both_players:
                score = score[:,0] if board.turn else score[:,1]
            return moves,score
        else:
            print(f'nodo terminal, resultado: {board.result()}')
            return None

    def get_move_values_single(self,board):
        t_moves=np.zeros([1,8,8,self.channels],dtype=np.float32)
        t_moves[0,:]=encode(board,self.encoding)
        score=self.model(t_moves)
        score=tf.nn.softmax(score,1)
        score=score.numpy()
        score = np.squeeze(score,0)
        return score
    


    def select_move(self,board):
        moves,values=self.get_move_values(board)
        index=np.argmax(values)
        return moves[index]

