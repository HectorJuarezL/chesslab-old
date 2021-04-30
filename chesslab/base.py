import time
import numpy as np
import pickle
import chess

class Tic:
    def __init__(self,type='minutes'):
        self.start=time.time()
        if type== 'minutes':
            self.type=1
        else:
            self.type=0
    def tic(self):
        self.start=time.time()
    def toc(self):
        if type==0:
            print('\nProcess finished - Elapsed time: {:.2f}s\n'.format(time.time()-self.start))
        else:
            print('\nProcess finished - Elapsed time: {:.2f}m\n'.format((time.time()-self.start)/60))




def load_pkl(filename):
    with open(filename, 'rb') as infile:
        return pickle.load(infile)

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
def order(x):
    return  {k: v for k, v in sorted(
        x.items(), key=lambda item: item[1], reverse=True)}


class default_parameters:
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




    

    np_encoding_1={
        0:np.array([0,0,0],dtype=np.float),
        1:np.array([0,0,1],dtype=np.float),
        2:np.array([0,0,-1],dtype=np.float),
        3:np.array([0,1,0],dtype=np.float),
        4:np.array([0,-1,0],dtype=np.float),
        5:np.array([1,0,0],dtype=np.float),
        6:np.array([-1,0,0],dtype=np.float),
        7:np.array([0,1,1],dtype=np.float),
        8:np.array([0,-1,-1],dtype=np.float),
        9:np.array([1,0,1],dtype=np.float),
        10:np.array([-1,0,-1],dtype=np.float),
        11:np.array([1,1,0],dtype=np.float),
        12:np.array([-1,-1,0],dtype=np.float)
    }

    np_encoding_2={
        0:np.array([0,0,0,0],dtype=np.float),
        1:np.array([1,0,0,0],dtype=np.float),
        2:np.array([0,0,0,1],dtype=np.float),
        3:np.array([0,1,0,0],dtype=np.float),
        4:np.array([0,0,1,0],dtype=np.float),
        5:np.array([1,1,0,0],dtype=np.float),
        6:np.array([0,0,1,1],dtype=np.float),
        7:np.array([1,0,1,0],dtype=np.float),
        8:np.array([0,1,0,1],dtype=np.float),
        9:np.array([1,0,0,1],dtype=np.float),
        10:np.array([0,1,1,0],dtype=np.float),
        11:np.array([1,1,1,0],dtype=np.float),
        12:np.array([0,1,1,1],dtype=np.float)
    }

    


    

