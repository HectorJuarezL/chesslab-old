import numpy as np
from .utils import params,load_pkl,save_pkl
import time

def preprocess(
    block_size=1000000,
    blocks=0,
    path='',
    start_name='',
    min_elo=3000,
    data_name='ccrl_states',
    labels_name='ccrl_results',
    elo_filter=1,
    nb_game_filter=0, #0 no aplica el filtro
    delete_eaten=True,
    delete_duplicate=True,
    delete_draws=True,
    delete_both_winners=True,
    undersampling = False):

    if blocks==0:
        print('specify the number of files to preprocess')
        return
    if path=='':
        print('specify the path where the files are stored')
        return

    start_time=time.time()

    #Reading data
    print('Reading blocks')
    
    state=np.zeros([blocks*block_size,64],np.int8)
    game=np.zeros([blocks*block_size],np.int32)

    if delete_draws:
        result=np.zeros([blocks*block_size,2],np.int8)
    else:
        result=np.zeros([blocks*block_size,3],np.int8)

    index_inf=0
    index_sup=0
    for i in range(1,blocks+1):
        print(f'file: {i}')
        temp_elo = load_pkl(f'{path}{start_name}_elo.{i}.pkl')
        temp_state = load_pkl(f'{path}{start_name}_state.{i}.pkl')
        temp_result = load_pkl(f'{path}{start_name}_result.{i}.pkl')
        temp_game = load_pkl(f'{path}{start_name}_game.{i}.pkl')

        if delete_draws:

            index_filter=np.where(temp_result[:,2]>0)
            temp_elo=np.delete(temp_elo,index_filter,0)
            temp_result=np.delete(temp_result,index_filter,0)
            temp_result=temp_result[:,:2]
            temp_state=np.delete(temp_state,index_filter,0)
            temp_game=np.delete(temp_game,index_filter,0)

        if elo_filter>0:

            #filtering by elo mean
            if elo_filter==1:
                b=np.mean(temp_elo,1)
            else: #filtering by elo min
                b=np.min(temp_elo,1)

            index_filter=np.where(b>=min_elo)[0]
       
            temp_state=temp_state[index_filter,:]
            temp_result=temp_result[index_filter,:]
            temp_game=temp_game[index_filter]

        index_inf = index_sup
        index_sup = index_sup+len(temp_game)
        
        state[index_inf:index_sup,:]=temp_state
        result[index_inf:index_sup,:]=temp_result
        game[index_inf:index_sup]=temp_game
    
    state=state[:index_sup,:]
    result=result[:index_sup,:]
    game=game[:index_sup]

    print('='*80)

    if delete_eaten:
        #en este bloque se eliminan aquellos estados que su estado siguiente sea comer una pieza, esto es porque muchos de estos estados conllevan el comer una pieza posterior
        print('Deleting states where there are eaten pieces')
        c=np.count_nonzero(state,1)
        print(c.shape)
        d=np.diff(c)
        print(d.shape)
        e=np.where(d==-1)
        print(e[0].shape)

        state=np.delete(state,e,0)
        result=np.delete(result,e,0)
        game=np.delete(game,e,0)

        len_state=len(state)
        len_result=len(result)
        len_game=len(game)
        print(f'total of different states: {len_state}')
        print(f'total of different result: {len_result}')
        print(f'total of games: {len_game}')
        print('='*80)


    #a continuación, se selecciona un número determinado de estados por juego
    if nb_game_filter>0:
        print(f'Selecting {nb_game_filter} game states per game')
        min_games=5
        max_games=nb_game_filter+min_games-1

        unique_games=len(np.unique(game))

        print(f'total of different games: {unique_games}')

        extracted_games=np.zeros([unique_games*nb_game_filter],dtype=np.int32) #guarda los indices de los juegos a extraer
        last_game=game[0]
        index_low=0
        cont=0
        cont_aux=0
        for i,g in enumerate(game):
            if g != last_game:
                if cont>max_games:
                    #extracted_games[cont_aux:cont_aux+nb_game_filter]=index_low+np.random.permutation((cont-min_games)//2)[:nb_game_filter]*2+min_games #this will get only nb_game_filter values from total per game
                    extracted_games[cont_aux:cont_aux+nb_game_filter]=index_low+np.arange(nb_game_filter)+min_games+np.random.randint(cont-min_games)
                    cont_aux+=nb_game_filter
                else:    
                    print(f'The game {g} has fewer turns than espected {cont}:{max_games}')
                last_game=g
                index_low=i
                cont=0
            cont+=1
        extracted_games[cont_aux:cont_aux+nb_game_filter]=index_low+np.random.permutation((len(game)-index_low-min_games)//2)[:nb_game_filter]*2+min_games

        if len(np.unique(extracted_games))<len(extracted_games): #this is just to verify that all index numbers are unique
            assert("extracted games error")

        state=state[extracted_games,:]
        result=result[extracted_games,:]
        game=game[extracted_games] #This variable is not longer needed

        len_state=len(state)
        len_result=len(result)
        print(f'total of different states: {len_state}')
        print(f'total of different results: {len_result}')
        print('='*80)


    if delete_duplicate:
        print('deleting duplicates')
        #Ahora, se eliminan los estados duplicados
        state_dict={}
        index=0
        result_aux=np.zeros(result.shape,dtype=np.int32)
        for i,key in enumerate(state):
            key=key.tobytes()
            if key not in state_dict:
                state_dict[key]=index
                index+=1
            result_aux[state_dict[key]]+=result[i]

        state=list(state_dict)
        result=result_aux[:index,:]

        del state_dict
        del result_aux

        np.max(result) #this is the reason why it is used uint16

        result[0] #just show a sample

        new_state=np.zeros((len(state),64),dtype=np.int8)
        for i,s in enumerate(state): new_state[i]=np.frombuffer(s,dtype=np.int8)

        state = new_state
        del new_state

        len_state=len(state)
        len_result=len(result)
        
        print(f'total of different states: {len_state}')
        print(f'total of different results: {len_result}')
        print('='*80)

    if delete_both_winners:

        #de los estados restantes, se eliminan aquellos que no tengan un claro ganador
        

        b=np.sum(result,axis=1)
        b=result/b[:,None]
        dif=np.abs(b[:,0]-b[:,1])
        a=np.where(dif<1)

        len(a[0])

        result=np.delete(b,a,0)
        result.shape

        np.max(result)

        result=result.astype(np.int) #now, the max value can be stored using 8-bits


        state=np.delete(state,a,0)
            
        len_state=len(state)
        len_result=len(result)
        print('deleting games with both winners')
        print(f'total of different states: {len_state}')
        print(f'total of different results: {len_result}')
        print('='*80)
    if undersampling:
        print('undersampling data')
        white_index=np.where(result[:,0]==1)
        white_index = np.random.permutation(white_index[0])
        white_states=state[white_index,:]
        white_results=result[white_index,:]
        del white_index

        black_index=np.where(result[:,1]==1)
        black_index = np.random.permutation(black_index[0])
        black_states=state[black_index,:]
        black_results=result[black_index,:]
        del black_index

        max_len=min(len(white_results),len(black_results))

        result = np.concatenate((white_results[:max_len,:],black_results[:max_len,:]))
        del white_results
        del black_results

        state = np.concatenate((white_states[:max_len,:],black_states[:max_len,:]))
        del white_states
        del black_states

        len_state=len(state)
        len_result=len(result)
        
        print(f'total of different states: {len_state}')
        print(f'total of different results: {len_result}')
        print('='*80)



    #se guardan los estados junto con sus etiquetas de ganador
    
    white_wins=np.count_nonzero(result[:,0]==1)
    black_wins=np.count_nonzero(result[:,1]==1)
    
    print("white total wins: {}".format(white_wins))
    print("black total wins: {}".format(black_wins))

    if black_wins>white_wins:
        print(f'IB={black_wins/white_wins:.2f}')
    else:
        print(f'IB={white_wins/black_wins:.2f}')


    print('saving files')
    
    save_pkl(state,f'{path}{data_name}.pkl')
    save_pkl(result,f'{path}{labels_name}.pkl')

    print('files saved')
    elapsed_time = time.time()-start_time
    print("Elapsed time: {:.0f}s = {:.1f}m".format(elapsed_time,elapsed_time/60))



