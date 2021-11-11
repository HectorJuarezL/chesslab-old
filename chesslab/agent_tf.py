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

    def __init__(self,model,path_model,cuda=True):
        physical_devices = tf.config.list_physical_devices('GPU')
        self.device = "/device:GPU:0" if cuda and len(physical_devices)>0 else "/cpu:0"
        print(self.device)
        self.encoding,self.history=load_model(model,path_model)
        self.model=model
        self.channels=len(self.encoding['.'])

    def get_moves_encoded_depth_1(self,board):
        moves=list(board.legal_moves)
        t_moves=np.zeros([len(moves),8,8,self.channels],dtype=np.float32)
        for i,m in enumerate(moves):
            board.push(m)
            t_moves[i,:]=encode(board,self.encoding)
            board.pop()
        return t_moves

    def get_moves_encoded_depth_2(self,board,parent_index):
        moves_1=list(board.legal_moves)
        moves_2=[]
        t_moves_1=np.zeros([len(moves),8,8,self.channels],dtype=np.float32)
        t_moves_2=np.zeros([0,8,8,self.channels],dtype=np.float32)
        parent_moves = np.zeros([0,],dtype=np.int32)
        for i,m in enumerate(moves):
            board.push(m)
            t_moves_1[i,:]=encode(board,self.encoding)
            moves,encoded = self.get_moves_encoded_depth_1(board)
            moves_2.extend(moves)
            t_moves_2 = np.concatenate((t_moves_2,encoded))
            parent_moves = np.concatenate((parent_moves,i+parent_index))
            board.pop()
        return moves_2,t_moves_2

        
    def get_move_values(self,board,both_players = False):
        moves=list(board.legal_moves)

        if len(moves)>0:
            t_moves=np.zeros([len(moves),8,8,self.channels],dtype=np.float32)
            for i,m in enumerate(moves):
                board.push(m)
                t_moves[i,:]=encode(board,self.encoding)
                board.pop()
            with tf.device(self.device):
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

