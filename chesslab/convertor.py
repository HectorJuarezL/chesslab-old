import time
import chess
import chess.pgn
import pickle
import sys
import numpy as np
import argparse

from .utils import Tic

def convert_games(source='',save_path='',start_name='chess',block_size=1000000,blocks=0,inter_map=None):
    if source=='':
        print('There is not source specified')
        return

    game_nb=np.zeros([block_size],dtype=np.int32)
    turn_nb=np.zeros([block_size],dtype=np.int16)
    state=np.zeros([block_size,64],dtype=np.int8)
    result=np.zeros([block_size,3],dtype=np.int8)
    elo=np.zeros([block_size,2],dtype=np.int16)
 
    pgn = open(source)
    game = chess.pgn.read_game(pgn)


    i=0
    cont=1
    nb=0
    
    tic=Tic()
    while game: 
        

        try:
            temp_elo=[game.headers['WhiteElo'],game.headers['BlackElo']]

            result_str = game.headers['Result']
            nb+=1
            j=0
                
            sys.stdout.write(f'\r {cont} block reading: {100*i/block_size:.2f}%')
            sys.stdout.flush()
            board = game.board()
            moves=list(game.mainline_moves())
            
            if '1-0' in result_str: #White wins
                winner=[1,0,0]
            elif '0-1' in result_str: #Black wins
                winner=[0,1,0]
            else:
                winner=[0,0,1]
            for v in moves:
                board.push(v) 
                b=str(board).replace(' ','').replace('\n','')
                d=np.array([inter_map[i] for i in list(b)],dtype=np.int8)

                state[i]=d
                result[i]=winner
                game_nb[i]=nb
                turn_nb[i]=j
                elo[i]=temp_elo
                
                i+=1
                j+=1
                if j>32767:
                    print('Hay partidas con más de 32767 movimientos, por lo que no es posible guardar el turno con 16 bits')
                    return

                if i%block_size == 0:
                    
                    i=0

                    with open(f'{save_path}{start_name}_game.{cont}.pkl', 'wb') as outfile:
                        pickle.dump(game_nb, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(f'{save_path}{start_name}_turn.{cont}.pkl', 'wb') as outfile:
                        pickle.dump(turn_nb, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(f'{save_path}{start_name}_state.{cont}.pkl', 'wb') as outfile:
                        pickle.dump(state, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(f'{save_path}{start_name}_elo.{cont}.pkl', 'wb') as outfile:
                        pickle.dump(elo, outfile, pickle.HIGHEST_PROTOCOL)
                    with open(f'{save_path}{start_name}_result.{cont}.pkl', 'wb') as outfile:
                        pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)


                    sys.stdout.write(f'\r {cont} block reading: 100.00%')
                    tic.toc()
                    if cont==blocks:
                        return
                    cont+=1
                    tic.tic()
        except KeyError:
            pass

            
        game = chess.pgn.read_game(pgn)
    
    
    game_nb=game_nb[:i]
    turn_nb=turn_nb[:i]
    state=state[:i,:]
    elo=elo[:i,:]
    result=result[:i,:]
    
    with open(f'{save_path}{start_name}_game.{cont}.pkl', 'wb') as outfile:
        pickle.dump(game_nb, outfile, pickle.HIGHEST_PROTOCOL)
    with open(f'{save_path}{start_name}_turn.{cont}.pkl', 'wb') as outfile:
        pickle.dump(turn_nb, outfile, pickle.HIGHEST_PROTOCOL)
    with open(f'{save_path}{start_name}_state.{cont}.pkl', 'wb') as outfile:
        pickle.dump(state, outfile, pickle.HIGHEST_PROTOCOL)
    with open(f'{save_path}{start_name}_elo.{cont}.pkl', 'wb') as outfile:
        pickle.dump(elo, outfile, pickle.HIGHEST_PROTOCOL)
    with open(f'{save_path}{start_name}_result.{cont}.pkl', 'wb') as outfile:
        pickle.dump(result, outfile, pickle.HIGHEST_PROTOCOL)

        
    tic.toc()
