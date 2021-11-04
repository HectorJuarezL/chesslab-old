from flask import Flask, render_template
from chesslab.agent_torch import agent
from chesslab.agent_mcts import agent_MCTS
import chess

import torch.nn as nn
class Model_1(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.func_1=nn.ELU()
        self.func_2=nn.ELU()
        self.func_3=nn.ELU()
        self.func_4=nn.ELU()
        
        self.cnn_1 = nn.Conv2d(3, 32, kernel_size=7,padding=3)
        self.cnn_2 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
        self.cnn_3 = nn.Conv2d(64, 128, kernel_size=3,padding=1)

        self.linear_1 = nn.Linear(8*8*128,256 )
        self.linear_2 = nn.Linear(256, 2)

    def forward(self, x ):
        out = self.cnn_1(x)
        out = self.func_1(out)
        out = self.cnn_2(out)
        out = self.func_2(out)
        out = self.cnn_3(out)
        out = self.func_3(out)
        out = out.reshape([x.size(0), -1])
        out = self.linear_1(out)
        out = self.func_4(out)
        out = self.linear_2(out)

        return out
model = Model_1()

deepbot = agent(model,'models/test_elo.0.3.pt',cuda=False)
deepMCTS = agent_MCTS(temperature=2,bot=deepbot,max_iter=1000,verbose = 2)
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
        move = str(deepMCTS.select_move(game_state,thinking_time=depth))
        print("Move found!", move)
        print()
        return move
    return "None"


@app.route('/test/<string:tester>')
def test_get(tester):
    return tester


if __name__ == '__main__':
    app.run()
    #app.run(host='0.0.0.0',port=8080)