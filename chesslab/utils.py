import time
import numpy as np
import sys
import os

version = sys.version
version = int(version[0]+version[2])
if version<=37:
    import pickle5 as pickle
else:
    import pickle


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            try:
                import google.colab
                return True
            except:
                return False # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


is_notebook = isnotebook()
if is_notebook:
    import IPython

class Print_r:
    def __init__(self,):
        if is_notebook:
            self.out = display(IPython.display.Pretty(''), display_id=True)
    def __call__(self,text):
        if is_notebook:
            self.out.update(IPython.display.Pretty(text))
        else:
            print('{}           \r'.format(text))


#def print_r(text="",new=False):
#    if is_notebook:
#    else:
#        print('{}           \r'.format(text))
#        #sys.stdout.write('{}           \r'.format(text))


def load_pkl(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

def save_pkl(data,filename):
    folder = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, 'wb') as pfile:
        pickle.dump(data, pfile, protocol=pickle.HIGHEST_PROTOCOL)

def create_csv(name,data,headers=None):
    file = open(name,'w',encoding='utf8')
    cols=len(data[0])
    template='{},'*(cols-1)+'{}\n'
    text=''
    if headers is not None:
        text=template.format(*headers)
    for row in data:
        text+=template.format(*row)
    file.write(text[:-1])

#funcion para ordenar un diccionario por su valor de mayor a menor
def order_dict(x,reverse=True):
    from operator import itemgetter
    return  {k: v for k, v in sorted(
        x.items(), key=itemgetter(1), reverse = reverse)}

def order_by_col(x,col=None,reverse=True):
    x.sort(key=itemgetter(col), reverse = reverse)
    return x

def get_max_row(x,col=None):
    return max(x,key=itemgetter(col))

def join_and_sort(moves,values):
    index = np.random.permutation(len(values)) #random permutation so same values can be sorted different
    values=values[index]
    moves=np.array(moves)
    moves=moves[index]

    index=np.argsort(max(values)-values)# the values are between 0 and 1, max(values)-values helps to get inverse sort
    #index=np.argsort(values)[::-1]

    values=values[index]
    moves=moves[index]
    values=np.expand_dims(values,axis=1)
    moves=np.expand_dims(moves,axis=1)
    return np.concatenate((moves,values),axis=1)

def download_file_from_google_drive(id, destination):
    import requests
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_7z(file_id,destination="./tmp/"):
    import py7zr
    tmp_name = 'temp.7z'
    download_file_from_google_drive(file_id, tmp_name)
    archive = py7zr.SevenZipFile(tmp_name, mode='r')
    archive.extractall(path=destination)
    archive.close()


class params:
    #number of game states extracted per block in convertor
    block_size=1000000
    batch_size=128
    test_size=0.1


    database_path = 'D:/Database_encoded/ccrl/'

    #variable to intemediate encoding
    inter_map={
        '.':0,
        'p':1,
        'P':2,
        'b':3,
        'B':4,
        'n':5,
        'N':6,
        'r':7,
        'R':8,
        'q':9,
        'Q':10,
        'k':11,
        'K':12,
    }

    encoding_1={
        '.':np.array([ 0, 0, 0],dtype=np.float),
        'p':np.array([ 0, 0, 1],dtype=np.float),
        'P':np.array([ 0, 0,-1],dtype=np.float),
        'b':np.array([ 0, 1, 0],dtype=np.float),
        'B':np.array([ 0,-1, 0],dtype=np.float),
        'n':np.array([ 1, 0, 0],dtype=np.float),
        'N':np.array([-1, 0, 0],dtype=np.float),
        'r':np.array([ 0, 1, 1],dtype=np.float),
        'R':np.array([ 0,-1,-1],dtype=np.float),
        'q':np.array([ 1, 0, 1],dtype=np.float),
        'Q':np.array([-1, 0,-1],dtype=np.float),
        'k':np.array([ 1, 1, 0],dtype=np.float),
        'K':np.array([-1,-1, 0],dtype=np.float)
    }

    encoding_2={
        '.':np.array([0,0,0,0],dtype=np.float),
        'p':np.array([1,0,0,0],dtype=np.float),
        'P':np.array([0,0,0,1],dtype=np.float),
        'b':np.array([0,1,0,0],dtype=np.float),
        'B':np.array([0,0,1,0],dtype=np.float),
        'n':np.array([1,1,0,0],dtype=np.float),
        'N':np.array([0,0,1,1],dtype=np.float),
        'r':np.array([1,0,1,0],dtype=np.float),
        'R':np.array([0,1,0,1],dtype=np.float),
        'q':np.array([1,0,0,1],dtype=np.float),
        'Q':np.array([0,1,1,0],dtype=np.float),
        'k':np.array([1,1,1,0],dtype=np.float),
        'K':np.array([0,1,1,1],dtype=np.float)
    }

def encode(board,inter_map=None):
    if inter_map == None:
        inter_map=params.inter_map
    b=str(board).replace(' ','').replace('\n','')
    return np.array([inter_map[i] for i in list(b)],dtype=np.int8)


def explore(board, depth):
    if depth == 0:
        return [board]
    
    moves = list(b.legal_moves)
    boards = []
    for m in moves:
        b=board.copy().push(m)
        boards.extend(explore(b,depth-1))
    return boards
        

