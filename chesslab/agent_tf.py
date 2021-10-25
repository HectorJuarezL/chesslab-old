import chess
import numpy as np
import tensorflow as tf
from .training_tf import load_model,encode


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


    def select_move(self,board):
        moves,values=self.get_move_values(board)
        index=np.argmax(values)
        return moves[index]

