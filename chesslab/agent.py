import chess
import numpy as np
from chesslab import base
from chesslab.base import default_parameters as params
from chesslab.training import torch_architectures as ta
import torch
from abc import ABC, abstractmethod

class Agent:
    
    @abstractmethod
    def get_move_values(self,board):
        pass

    @abstractmethod
    def select_move(self,board):
        pass


class random_agent(Agent):
    def get_move_values(self,board):
        pass
    def select_move(self,board):
        moves=list(board.legal_moves)
        if len(moves)>0:
            return moves[np.random.randint(len(moves))]
        else:
            return []

class chess_torch_agent(Agent):
    def __init__(self,path_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model,self.encoding,self.history=ta.load_model(path_model)
        self.model.to(self.device)
        self.model.eval()
        self.channels=len(self.encoding['.'])
        
    
    def get_move_values(self,board):
        moves=list(board.legal_moves)

        if len(moves)>0:
            with torch.no_grad():
                t_moves=torch.zeros([len(moves),self.channels,8,8],dtype=torch.float,device=self.device)
                for i,m in enumerate(moves):
                    board.push(m)
                    t_moves[i,:]=ta.encode(board,self.encoding)
                    board.pop()
                score=torch.softmax(self.model(t_moves),1)
                
                score=score.tolist()
                return list(map(list,zip(moves,score)))
        else:
            print(f'nodo terminal, resultado: {board.result()}')
            return []
    def select_move(self,board):
        values=self.get_move_values(board)
        if board.turn:
            result={move:valor for move,(valor,_) in values}
        else:
            result={move:valor for move,(_,valor) in values}
        result=base.order(result)
        return list(result.keys())[0]


