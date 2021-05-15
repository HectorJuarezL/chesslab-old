
import sys
import pickle

import datetime as dt
import time
import numpy as np

from chesslab.base import load_pkl
from chesslab.base import default_parameters as params


from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model
import tensorflow as tf


class keras_architectures:
    lr = 0.1
    mo = 0.1


    def train(start=0,epochs=1,batch_size=128,x_train=None,y_train=None,x_test=None,y_test=None,
    	model=None, save_name='model',encoding=None,load_model=None):

       
        #NUM_EPOCHS = start+epochs

        #inter_encoding = {params.inter_map[i]:code for i,code in encoding.items()}


        history=model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(x_test,y_test))


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

    def recode(x_in,encoding):
        to_return=np.zeros([x_in.shape[0],64,len(encoding[0])])
        for key,value in encoding.items():
            to_change=np.where(x_in==key)
            to_return[to_change[0],to_change[1],:]=value
        return to_return.reshape([-1,8,8,len(encoding[0])])

    def encode(board,encoding):
        b=str(board).replace(' ','').split('\n')
        a=torch.zeros([8,8,len(encoding['.'])])
        for i,row in enumerate(b):
            for j,val in enumerate(row):
                a[i,j,:]=encoding[val]
        return a


    Model_1 = tf.keras.Sequential([
      layers.Input(shape=( 8, 8, 3)), 
      layers.Conv2D(32, 7, activation='elu', padding='same'),
      layers.Conv2D(64, 5, activation='elu', padding='same'),
      layers.Conv2D(128, 3, activation='elu', padding='same'),
      layers.Flatten(),
      layers.Dense(128, activation='elu'),
      layers.Dense(2, activation='softmax')])

    Model_2 = tf.keras.Sequential([
      layers.Input(shape=( 8, 8, 3)), 
      layers.Conv2D(32, 7, activation='elu', padding='same'),
      layers.Conv2D(64, 5, activation='elu', padding='same'),
      layers.Conv2D(128, 3, activation='elu', padding='same'),
      layers.Conv2D(32, 1, activation='elu', padding='same'),
      layers.Conv2D(8, 1, activation='elu', padding='same'),
      layers.Flatten(),
      layers.Dense(64, activation='elu'),
      layers.Dense(2, activation='softmax')])

