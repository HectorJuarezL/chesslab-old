import chess
import chess.engine
import numpy as np


class agent_stockfish():

    def __init__(self,path):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)  
        
    
    def get_move_values(self,board,depth=0,both_players = False):
        moves=list(board.legal_moves)
        score = np.zeros((len(moves),))
        if len(moves)>0:
                for i,m in enumerate(moves):
                    board.push(m)
                    info = self.engine.analyse(board, chess.engine.Limit(depth=depth), info=chess.engine.INFO_SCORE)
                    s = info['score'].relative.score()
                    if s is not None:
                        score[i]=s
                    board.pop()
                #min_score = np.min(score)
                if board.turn:
                    score = -score
                score = score-np.min(score)
                score = score/np.max(score)
                if both_players:
                    score = np.concatenate((score.reshape((-1,1)),1-score.reshape((-1,1))),axis=1)
                return moves,score
        else:
            print(f'nodo terminal, resultado: {board.result()}')
            return []


    def select_move(self,board):
        moves,values=self.get_move_values(board)
        index=np.argmax(values)
        return moves[index]

