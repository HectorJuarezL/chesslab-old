import chess
import numpy as np
import torch
from .training_torch import load_model,encode


class agent():

    def __init__(self,model,path_model,cuda=True):
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        print('device: {}'.format(self.device))
        self.model,self.encoding,self.history=load_model(model,path_model,device=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.channels=len(self.encoding['.'])
        
    def get_move_values(self,board,both_players = False):
        moves=list(board.legal_moves)

        if len(moves)>0:
            with torch.no_grad():
                t_moves=torch.zeros([len(moves),self.channels,8,8],dtype=torch.float32,device=self.device)
                for i,m in enumerate(moves):
                    board.push(m)
                    t_moves[i,:]=encode(board,self.encoding)
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

