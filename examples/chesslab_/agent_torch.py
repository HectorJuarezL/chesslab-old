import chess
import numpy as np
from .utils import default_parameters as params
import torch
import torch.nn as nn
from .training_torch import training as ta



class agent():

    def __init__(self,path_model,model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.model,self.encoding,self.history=ta.load_model(path_model,model)
        self.model.to(self.device)
        self.model.eval()
        self.channels=len(self.encoding['.'])
        
    
    def get_move_values(self,board,both_players = False):
        moves=list(board.legal_moves)

        if len(moves)>0:
            with torch.no_grad():
                t_moves=torch.zeros([len(moves),self.channels,8,8],dtype=torch.float,device=self.device)
                for i,m in enumerate(moves):
                    board.push(m)
                    t_moves[i,:]=ta.encode(board,self.encoding)
                    board.pop()
                score=self.model(t_moves).cpu()
                score=torch.softmax(score,1)
                score=score.detach().numpy()
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

