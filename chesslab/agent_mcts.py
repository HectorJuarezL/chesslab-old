import chess
import numpy as np
import time


class MCTSNode:
    def __init__(self, game_state, parent = None, move = None, value = [0.5,0.5], bot = None, isRoot = False,scale_factor=10):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = np.zeros([2])
        self.value=value
        self.num_rollouts = 0
        self.children = []
        self.scale_factor=scale_factor
        if not self.is_terminal():
            self.unvisited_moves,self.unvisited_values = bot.get_move_values(game_state,both_players=True)
            self.unvisited_values = self.unvisited_values.tolist()
        else:
            tmp = game_state.result()
            self.win_counts[0] = int(tmp[0])
            self.win_counts[1] = int(tmp[2])

            if self.win_counts[0]==0 or self.win_counts[1]==0:
                self.win_counts*=scale_factor
            self.unvisited_moves = []
            self.unvisited_values = []
        self.isRoot=isRoot

    def add_random_child(self,bot):
        index = np.random.randint(len(self.unvisited_moves))
        new_move = self.unvisited_moves.pop(index)#selecciona un movimiento disponible al azar y lo elimina de los movimientos no visitados
        new_value = self.unvisited_values.pop(index)
        new_game_state = self.game_state.copy(stack=False) #crea una copia del estado de juego
        
        new_game_state.push(new_move) #realiza el movimiento seleccionado
        new_node = MCTSNode(game_state=new_game_state, parent=self, move=new_move,value=new_value,bot=bot,scale_factor=self.scale_factor) #crea un nuevo nodo
        self.children.append(new_node) #añade el nodo a su lista de hijos
        return new_node #retorna el nuevo nodo

    def record_win(self, result):
        self.win_counts = self.win_counts+[1,0] if np.random.rand()<result[0] else self.win_counts+[0,1]
        #self.win_counts += result
        self.num_rollouts += 1

    def can_add_child(self): #comprueba si aun hay nodos por visitar
        return len(self.unvisited_moves) > 0

    def is_terminal(self): #verifica si es un nodo terminal, es decir, el final de una partida
        return self.game_state.is_game_over()

    def winning_frac(self, player): #obtiene el valor Q/N para el nodo dado
        if player: #turno de las blancas
            return float(self.win_counts[0]) / float(self.num_rollouts)
        else: #turno de las negras
            return float(self.win_counts[1]) / float(self.num_rollouts)

class agent_MCTS:
    def __init__(self, temperature=None,bot=None,game_state=None,max_iter=100):
        self.temperature = temperature
        self.bot = bot
        self.max_iter = max_iter
        self.root = MCTSNode(game_state.copy(stack=False),bot=self.bot,isRoot=True)

    def select_move(self,board,max_iter=None,both_players=False):
        moves,values=self.get_move_values(board,max_iter=max_iter,both_players=both_players)
        index=np.argmax(values)
        self.push_move(moves[index])
        return moves[index]
        
    def push_move(self,move):
        root=self.root
        for child in root.children:
            if child.move==move:
                child.isRoot=True
                self.root=child
                return
        print("Error, movimiento no existente")
        return

    def set_max_iter(self,max_iter=100):
        self.max_iter=max_iter

    def select_child(self, node):
        """
            Selecciona un hijo usando la métrica UCT (Upper confidence bound for trees).
        """

        #Calcula N(v)
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = np.log(total_rollouts)

        best_score = -1
        best_child = None
        #Calcula UTC(j)
        for child in node.children:
            #win_percentage = child.winning_frac(root.game_state.turn)
            win_percentage = child.winning_frac(node.game_state.turn)
            exploration_factor = np.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def get_move_values(self,game_state,max_iter=None,both_players = False):
        
        if max_iter is None:
            max_iter=self.max_iter

        if str(self.root.game_state)!=str(game_state):
            #print('\nEl estado de juego no corresponde con el de la raiz del arbol, se recreó la raiz')
            self.root = MCTSNode(game_state,bot=self.bot,isRoot=True)

        root=self.root
        #print("\n")
        i=0

        tic = time.time()
        while i<max_iter:
            #print(i,end=" ")
            i+=1
            node = root
            #fase de seleccion, donde busca un nodo que no sea un nodo derminal
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            #fase de expansión, donde se agrega un nuevo nodo
            if node.can_add_child():
                node = node.add_random_child(self.bot)

            #fase de simulación. Con ayuda de la red neuronal, se obtiene el valor del nodo que predice como ganador
            winner = node.value

            #fase de retropropagación, donde se actualiza el valor de Q de los nodos padres hasta llegar al nodo raiz
            while node is not None:
                node.record_win(winner)
                node = node.parent
        toc = time.time()-tic
        print('MCTS - Elapsed time: {:.2f}s = {:.2f}m'.format(toc,toc/60))

        
        score = np.zeros(len(root.children),)
        moves = []
        for i,child in enumerate(root.children):
            score[i]=child.winning_frac(root.game_state.turn)
            moves.append(child.move)
        if both_players:
            score = np.concatenate((score.reshape((-1,1)),1-score.reshape((-1,1))),axis=1)
        return moves,score

