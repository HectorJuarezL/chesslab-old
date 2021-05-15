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

import requests

def download_file_from_google_drive(id, destination):
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

    encoding_1={
        '.':np.array([0,0,0],dtype=np.float),
        'p':np.array([0,0,1],dtype=np.float),
        'P':np.array([0,0,-1],dtype=np.float),
        'b':np.array([0,1,0],dtype=np.float),
        'B':np.array([0,-1,0],dtype=np.float),
        'n':np.array([1,0,0],dtype=np.float),
        'N':np.array([-1,0,0],dtype=np.float),
        'r':np.array([0,1,1],dtype=np.float),
        'R':np.array([0,-1,-1],dtype=np.float),
        'q':np.array([1,0,1],dtype=np.float),
        'Q':np.array([-1,0,-1],dtype=np.float),
        'k':np.array([1,1,0],dtype=np.float),
        'K':np.array([-1,-1,0],dtype=np.float)
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
        inter_map=default_parameters.inter_map
    b=str(board).replace(' ','').replace('\n','')
    return np.array([inter_map[i] for i in list(b)],dtype=np.int8)


    

