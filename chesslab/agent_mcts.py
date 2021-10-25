import chess
import numpy as np
import time
from numpy.random import default_rng
rng = default_rng()

class MCTS_graph:
    def __init__(self,agent):
        self.root=agent.root
        self.temperature = agent.temperature
    def make_graph(self,depth=1000):
        self.cont=0
        self.nodes = {}
        self.edges = []

        self.bfs(self.root,0,depth)
        print('Total nodes: {}'.format(self.cont))
    def bfs(self,node,father,depth):
        if depth==0: return
        if len(node.children)>0:
            log_rollouts = np.log(node.num_rollouts)
            for n in node.children:
                self.cont+=1
                win_percentage = n.winning_frac()
                self.nodes[self.cont]=win_percentage
                self.edges.append([father,self.cont,n.move])
                self.bfs(n,self.cont,depth-1)

    def save_graph(self,path,depth=1000):
        with open(path,'w') as file:
            self.make_graph(depth)
            cad="digraph{\n  0 [label=\"root\"];\n"
            for n,m in self.nodes.items():
                cad+="  {} [label=\"{:.2f}\"];\n".format(n,m)
            for (x,y,z) in self.edges:
                cad+="  {} -- {} [label=\"{}\"];\n".format(x,y,z)
            cad+="}"
            file.write(cad)
            print("Grafo guardado en: {}".format(path))



class MCTSNode:
    def __init__(self, game_state, parent = None, move = None, value = [0,0], bot = None, is_root = False):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = np.zeros([2,])
        self.value=np.zeros([2,])
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = []
        self.unvisited_values = []
        self.is_root=is_root
        if self.is_terminal():
            tmp = game_state.result()
            if int(tmp[0]) == 0:
                self.value = [0,1]
            elif int(tmp[2]) == 0:
                self.value = [1,0]
            else:
                self.value = [1/2,1/2]
        else:
            self.value+=value
            self.unvisited_moves,self.unvisited_values = bot.get_move_values(game_state,both_players=True)
            self.unvisited_values = self.unvisited_values.tolist()
        

    def add_random_child(self,bot):
        index = np.random.randint(len(self.unvisited_moves))
        new_move = self.unvisited_moves.pop(index) #selecciona un movimiento disponible al azar y lo elimina de los movimientos no visitados
        new_value = self.unvisited_values.pop(index)
        new_game_state = self.game_state.copy(stack=False) #crea una copia del estado de juego
        new_game_state.push(new_move) #realiza el movimiento seleccionado
        new_node = MCTSNode(game_state=new_game_state, parent=self, move=new_move,value=new_value,bot=bot) #crea un nuevo nodo
        self.children.append(new_node) #añade el nodo a su lista de hijos
        return new_node #retorna el nuevo nodo

    def record_win(self, result, temperature=2):
        self.win_counts += result
        self.num_rollouts += 1
        if not self.is_root:
            log_rollouts = np.log(self.parent.num_rollouts+1)#se le suma 1 porque aun no se ha actualizado el valor del padre
            self.update_uct(log_rollouts,temperature)


    def result_simulation(self):
        return self.value

    def can_add_child(self): #comprueba si aun hay nodos por visitar
        return len(self.unvisited_moves) > 0

    def is_terminal(self): #verifica si es un nodo terminal, es decir, el final de una partida
        return self.game_state.is_game_over()

    def winning_frac(self): #obtiene el valor Q/N para el nodo dado
        if self.parent.game_state.turn: #turno de las blancas
            return float(self.win_counts[0]) / float(self.num_rollouts)
        else: #turno de las negras
            return float(self.win_counts[1]) / float(self.num_rollouts)

    def update_uct(self,log_rollouts,temperature=2):
        #log_rollouts = np.log(self.parent.num_rollouts)
        win_percentage = self.winning_frac()
        exploration_factor = np.sqrt(log_rollouts / self.num_rollouts)
        self.uct_score = win_percentage + temperature * exploration_factor

class agent_MCTS:
    def __init__(self, temperature=2,bot=None,game_state=None,max_iter=100,verbose=0):
        self.temperature = temperature
        self.bot = bot
        self.max_iter = max_iter
        self.root = None
        self.verbose = verbose
        if game_state is not None:
            self.root = MCTSNode(game_state.copy(),bot=self.bot,is_root=True)

    def select_move(self,board,max_iter=None,push=True):
        moves,values=self.get_move_values(board,max_iter=max_iter)
        if moves is None:
            return None
        index=np.argmax(values)
        if push:
            self.push_move(move=moves[index])
        return moves[index]
        
    def push_move(self,move=None):
        for child in self.root.children:
            if child.move==move:
                child.is_root=True
                self.root=child
                self.root.num_rollouts-=1
                self.root.parent=None
                return True
        return False

    def push_board(self,board=None):
        str_board=str(board)
        for child in self.root.children:
            if str(child.game_state) == str_board:
                child.is_root=True
                self.root=child
                self.root.num_rollouts-=1
                self.root.parent=None
                return True
        return False

    def set_max_iter(self,max_iter=100):
        self.max_iter=max_iter

    def select_child(self, node):
        #best_score = -1
        #best_child = None
        uct_score = np.array([child.uct_score for child in node.children])
        best_child = np.argmax(uct_score)
        return node.children[best_child]
        #for child in node.children:
        #    uct_score = child.uct_score
        #    if uct_score > best_score:
        #        best_score = uct_score
        #        best_child = child
        

    def get_move_values(self,game_state,max_iter=None):
        
        if max_iter is None:
            max_iter=self.max_iter

        if (self.root is None) or (str(self.root.game_state)!=str(game_state) and not self.push_board(board=game_state)):
            if self.verbose>0:
                print('\nEl estado de juego no corresponde con el de la raiz del arbol, se recreó la raiz')
            self.root = MCTSNode(game_state.copy(stack=False),bot=self.bot,is_root=True)

        if self.root.is_terminal():
            return None,None
        i=0

        tic = time.time()
        while i<max_iter:
            #print(i,end=" ")
            i+=1
            node = self.root
            #fase de seleccion, donde busca un nodo que no sea un nodo derminal
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            #fase de expansión, donde se agrega un nuevo nodo
            if node.can_add_child():
                node = node.add_random_child(self.bot)

            #fase de simulación. Con ayuda de la red neuronal, se obtiene el valor del nodo que predice como ganador
            result = node.result_simulation()

            #fase de retropropagación, donde se actualiza el valor de Q de los nodos padres hasta llegar al nodo raiz
            while node is not None:
                node.record_win(result,self.temperature)
                node = node.parent
        if self.verbose>1:
            toc = time.time()-tic
            print('MCTS - rollouts:{} Elapsed time: {:.2f}s = {:.2f}m'.format(self.root.num_rollouts,toc,toc/60))

        
        score = []
        moves = []
        for child in self.root.children:
            win_percentage=child.winning_frac()
            score.append(win_percentage)
            moves.append(child.move)
        score = np.array(score)
        return moves,score

