import time
import chess
import chess.pgn
import pickle
import sys
import numpy as np
import argparse
from chesslab.base import Tic
from chesslab.base import default_parameters as params



def convert_games(source='',save_path='',start_name='chess',block_size=1000000,blocks=0,inter_map=None):
    if source=='':
        print('There is not source specified')
        return

    game_nb=np.zeros([block_size],dtype=np.int32)
    turn_nb=np.zeros([block_size],dtype=np.int16)
    state=np.zeros([block_size,64],dtype=np.int8)
    result=np.zeros([block_size,2],dtype=np.int8)
    elo=np.zeros([block_size,2],dtype=np.int16)
 
    pgn = open(source)
    game = chess.pgn.read_game(pgn)


    i=0
    cont=1
    nb=0
    
    tic=Tic()
    while game: 
        result_str = game.headers['Result']
        nb+=1
        j=0
        
        if '1-0' in result_str or '0-1' in result_str:
        
            
            sys.stdout.write(f'\r {cont} block reading: {100*i/block_size:.2f}%')
            sys.stdout.flush()
            board = game.board()
            moves=list(game.mainline_moves())
            temp_elo=[game.headers['WhiteElo'],game.headers['BlackElo']]
            
            if '1-0' in result_str: #White wins
                winner=[1,0]
            elif '0-1' in result_str: #Black wins
                winner=[0,1]
            for v in moves:
                board.push(v) 
                b=str(board).replace(' ','').replace('\n','')
                d=np.array([inter_map[i] for i in list(b)],dtype=np.uint8)

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


                    sys.stdout.write(f'\r block reading: 100.00%')
                    tic.toc()
                    if cont==blocks:
                        return
                    cont+=1
                    tic.tic()

            
        game = chess.pgn.read_game(pgn)
    
    cont+=1
    
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Muestra los parámetros disponibles y un ejemplo de uso.')
    parser.add_argument('--block_size', type=int, default=params.block_size, help="Tamaño del bloque de estados a guardar por archivo")
    parser.add_argument('--blocks', type=int, default=0, help="Número total de bloques a extraer, 0=todos los que permita el archivo PGN")
    parser.add_argument('--start_name', type=str, default=params.start_name, help="Tamaño de la muestra a usar.")
    parser.add_argument('--source_file', type=str, default='', help="Archivo PGN a convertir")
    parser.add_argument('--save_path', type=str, default='', help="Direccion donde se guardará la base preprocesada")

    inter_map=params.inter_map

    args = parser.parse_args()

    tic=Tic()
    process_games(source=args.source_file,save_path=args.save_path,start_name=args.start_name,block_size=start.block_size,inter_map=inter_map)
    tic.toc()