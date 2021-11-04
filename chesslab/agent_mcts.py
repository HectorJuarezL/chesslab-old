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
    def __init__(self, game_state, parent = None, move = None, bot = None, is_root = False):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = np.zeros([2,])
        self.value=np.zeros([2,])
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = []
        self.is_root=is_root
        if self.is_terminal():
            tmp = game_state.result()
            if int(tmp[0]) == 0:
                self.value = np.array([0,1])
            elif int(tmp[2]) == 0:
                self.value = np.array([1,0])
            else:
                self.value = np.array([1/2,1/2])
        else:
            self.unvisited_moves = list(game_state.legal_moves)
            value = bot.get_move_values_single(game_state)
            self.value+=value
        

    def add_random_child(self,bot):
        index = np.random.randint(len(self.unvisited_moves))
        new_move = self.unvisited_moves.pop(index) #selecciona un movimiento disponible al azar y lo elimina de los movimientos no visitados
        new_game_state = self.game_state.copy(stack=False) #crea una copia del estado de juego
        new_game_state.push(new_move) #realiza el movimiento seleccionado
        new_node = MCTSNode(game_state=new_game_state, parent=self, move=new_move, bot=bot) #crea un nuevo nodo
        self.children.append(new_node) #añade el nodo a su lista de hijos
        return new_node #retorna el nuevo nodo

    def record_win(self, result):
        self.win_counts += result
        self.num_rollouts += 1


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

class agent_MCTS:
    def __init__(self, temperature=2,bot=None,game_state=None,max_iter=100,verbose=0):
        self.temperature = temperature
        self.bot = bot
        self.max_iter = max_iter
        self.root = None
        self.verbose = verbose
        if game_state is not None:
            self.root = MCTSNode(game_state.copy(),bot=self.bot,is_root=True)

    def select_move(self,board,max_iter=None,push=True, thinking_time = 0):
        moves,values=self.get_move_values(board,max_iter=max_iter, thinking_time = thinking_time)
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
        best_score = -1
        best_child = None
        log_rollouts = np.log(node.num_rollouts)
        for child in node.children:
            win_percentage = child.winning_frac()
            exploration_factor = np.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child
        

    def get_move_values(self,game_state,max_iter=None, thinking_time = 0):
        
        if max_iter is None:
            max_iter=self.max_iter

        if (self.root is None) or (str(self.root.game_state)!=str(game_state) and not self.push_board(board=game_state)):
            if self.verbose>0:
                print('El estado de juego no corresponde con el de la raiz del arbol, se recreó la raiz')
            self.root = MCTSNode(game_state.copy(stack=False),bot=self.bot,is_root=True)

        if self.root.is_terminal():
            return None,None
        i=0

        tic = time.time()
        while thinking_time>0 or i<max_iter:
            toc = time.time()-tic
            if toc> thinking_time:
                thinking_time=0
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
                node.record_win(result)
                node = node.parent
        if self.verbose>1:
            toc = time.time()-tic
            print('MCTS - nodes:{} Elapsed time: {:.2f}s = {:.2f}m nps={:.0f}'.format(self.root.num_rollouts,toc,toc/60,self.root.num_rollouts/toc))

        
        score = []
        moves = []
        for child in self.root.children:
            win_percentage=child.winning_frac()
            score.append(win_percentage)
            moves.append(child.move)
        score = np.array(score)
        return moves,score

