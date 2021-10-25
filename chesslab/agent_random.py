import chess
import numpy as np

class agent():
    def get_move_values(self,board):
        moves=list(board.legal_moves)

        if len(moves)>0:
            if len(moves)>1:
                score = np.random.permutation(len(moves))/(len(moves)-1)
            else:
                score = np.array([1])
            return moves,score
        else:
            print(f'nodo terminal, resultado: {board.result()}')
            return None


    def select_move(self,board):
        moves,values=self.get_move_values(board)
        index=np.argmax(values)
        return moves[index]