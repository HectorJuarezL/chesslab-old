from flask import Flask, render_template
from chesslab.agent_tf import agent
from chesslab.agent_mcts import agent_MCTS
import chess

import numpy as np

import tensorflow as tf
class Model_1():
    def __init__(self,
                   n_classes=2):
        initializer = tf.keras.initializers.GlorotNormal()
        self.hw=[]
        self.hb=[]

        self.hw.append( tf.Variable(initializer(shape=(7,7,3,32)),name="hl1weigths",dtype="float32") )
        self.hb.append( tf.Variable(np.zeros(32),name="hl1bias",dtype="float32") )
        #8x8x32
        self.hw.append( tf.Variable(initializer(shape=(5,5,32,64)),name="hl2weigths",dtype="float32"))
        self.hb.append( tf.Variable(np.zeros(64),name="hl2bias",dtype="float32"))
        #8x8x64
        self.hw.append( tf.Variable(initializer(shape=(3,3,64,128)),name="hl3weigths",dtype="float32"))
        self.hb.append( tf.Variable(np.zeros(128),name="hl3bias",dtype="float32"))
        #8x8x128
        self.hw.append( tf.Variable(initializer(shape=(8*8*128,256)),name="hl4weigths",dtype="float32"))
        self.hb.append( tf.Variable(np.zeros(256),name="hl4bias",dtype="float32"))

        self.hw.append( tf.Variable(initializer(shape=(256, n_classes)),name="outweigths",dtype="float32"))
        self.hb.append( tf.Variable(np.zeros(n_classes),name="outbias",dtype="float32"))

        self.trainable_variables = []
        for i in range(len(self.hw)):
            self.trainable_variables.append(self.hw[i])    
            self.trainable_variables.append(self.hb[i])

    def __call__(self,x): 

        # Declarando la arquitectura
        out = tf.cast(x, tf.float32)
        out = tf.reshape(out, shape=[-1, 8, 8, 3])

        layer=0
        out = tf.add(out, 1e-8)
        out = tf.nn.conv2d(out,self.hw[layer], strides=[1,1,1,1], padding='SAME')  
        out = tf.add(out, self.hb[layer])
        out = tf.nn.elu(out)
              #8*8*32
        layer+=1
        out = tf.nn.conv2d(out,self.hw[layer], strides=[1,1,1,1], padding='SAME')  
        out = tf.add(out, self.hb[layer])
        out = tf.nn.elu(out)
              #8*8*64
        layer+=1  
        out = tf.nn.conv2d(out,self.hw[layer], strides=[1,1,1,1], padding='SAME')  
        out = tf.add(out, self.hb[layer])
        out = tf.nn.elu(out)
              #8*8*128
        layer+=1
        out =  tf.reshape(out,[-1, 8*8*128])
        out =  tf.matmul(out,self.hw[layer])
        out = tf.add(out, self.hb[layer])
        out = tf.nn.elu(out)

        layer+=1
        out =  tf.matmul(out,self.hw[layer])
        out = tf.add(out, self.hb[layer])

        return out
model = Model_1()

deepbot = agent(model,'models/tf_elo.0.5.h5')
deepMCTS = agent_MCTS(temperature=2,bot=deepbot,max_iter=1000)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/move/<int:depth>/<path:fen>/')
def get_move(depth, fen):
    print(depth)
    print("Calculating...")
    game_state=chess.Board(fen)
    if not game_state.is_game_over():
        move = str(deepMCTS.select_move(game_state))
        print("Move found!", move)
        print()
        return move
    return "None"


@app.route('/test/<string:tester>')
def test_get(tester):
    return tester


if __name__ == '__main__':
    app.run()