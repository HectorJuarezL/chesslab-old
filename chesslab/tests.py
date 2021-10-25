from .utils import join_and_sort,Print_r
import numpy as np
import chess


def versus(agent_white=None,agent_black=None,n_counts=100):
    white_wins=0
    black_wins=0
    draws =0
    print_r = Print_r()
    percent = n_counts//100
    for i in range(n_counts):
        if percent<100 or i%percent==0:
            print_r("Progress:{:.0f}/100 | White wins: {} | Black wins: {} | Draws:{}".format(i/n_counts*100, white_wins,black_wins,draws))
        board = chess.Board()
        while not board.is_game_over():
            move=agent_white.select_move(board)
            board.push(move)
            if not board.is_game_over():
                move=agent_black.select_move(board)
                board.push(move)
        result = board.result()        
        if int(result[0]) == 0:
            black_wins+=1
        elif int(result[2]) == 0:
            white_wins+=1
        else:
            draws+=1
    print_r("Progress:{:.0f}/100 | White wins: {} | Black wins: {} | Draws:{}".format(100, white_wins,black_wins,draws))
        
    #white_wins/=n_counts
    #black_wins/=n_counts
    #draws/=n_counts
    return white_wins,black_wins,draws


def kaufman_test(agent,details=0,verbose=0):
    acc=0
    acc_w=0
    if details>0:
        data=[]
        if details==2:
            boards=[]
    if verbose>0:
        print("Kaufman test ")
    for i,s in kaufman_states.items():
        if verbose>0:
            print(i)
        board,y = chess.Board.from_epd(s)
        #y=board.set_epd(s)
        bm=list(y.values())[0][0]
        if details==0:
            move=agent.select_move(board)
            if move == bm:
                acc+=1
        elif details==1 or details==2:
            moves,values=agent.get_move_values(board)
            sort=join_and_sort(moves,values)
            bm_predicted=sort[0,0]
            value_predicted=sort[0,1]
            rank=np.where(sort[:,0]==bm)[0][0]
            value_bm=sort[rank,1]
            weight=1-rank/(len(moves)-1)
            tmp=[bm,value_bm,bm_predicted,value_predicted,rank+1,len(moves),weight]
            acc_w+=weight
            data.append(tmp)
            if details==2:
                boards.append(board.copy(stack=False))
            if bm_predicted == bm:
                acc+=1
        else:
            print("detail not valid")
            
        
    acc/=25
    acc_w/=25
    
    if details==1:
        return acc,acc_w,data
    elif details==2:
        return acc,acc_w,boards,data
    else:
        return acc


kaufman_states={
    1:"1rbq1rk1/p1b1nppp/1p2p3/8/1B1pN3/P2B4/1P3PPP/2RQ1R1K w - - bm Nf6+",
    2:"3r2k1/p2r1p1p/1p2p1p1/q4n2/3P4/PQ5P/1P1RNPP1/3R2K1 b - - bm Nxd4",
    3:"3r2k1/1p3ppp/2pq4/p1n5/P6P/1P6/1PB2QP1/1K2R3 w - - am Rd1",
    4:"r1b1r1k1/1ppn1p1p/3pnqp1/8/p1P1P3/5P2/PbNQNBPP/1R2RB1K w - - bm Rxb2",
    5:"2r4k/pB4bp/1p4p1/6q1/1P1n4/2N5/P4PPP/2R1Q1K1 b - - bm Qxc1",
    6:"r5k1/3n1ppp/1p6/3p1p2/3P1B2/r3P2P/PR3PP1/2R3K1 b - - am Rxa2",
    7:"2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - bm Bxe4",
    8:"5r1k/6pp/1n2Q3/4p3/8/7P/PP4PK/R1B1q3 b - - bm h6",
    9:"r3k2r/pbn2ppp/8/1P1pP3/P1qP4/5B2/3Q1PPP/R3K2R w KQkq - bm Be2",
    10:"3r2k1/ppq2pp1/4p2p/3n3P/3N2P1/2P5/PP2QP2/K2R4 b - - bm Nxc3",
    11:"q3rn1k/2QR4/pp2pp2/8/P1P5/1P4N1/6n1/6K1 w - - bm Nf5",
    12:"6k1/p3q2p/1nr3pB/8/3Q1P2/6P1/PP5P/3R2K1 b - - bm Rd6",
    13:"1r4k1/7p/5np1/3p3n/8/2NB4/7P/3N1RK1 w - - bm Nxd5",
    14:"1r2r1k1/p4p1p/6pB/q7/8/3Q2P1/PbP2PKP/1R3R2 w - - bm Rxb2",
    15:"r2q1r1k/pb3p1p/2n1p2Q/5p2/8/3B2N1/PP3PPP/R3R1K1 w - - bm Bxf5",
    16:"8/4p3/p2p4/2pP4/2P1P3/1P4k1/1P1K4/8 w - - bm b4",
    17:"1r1q1rk1/p1p2pbp/2pp1np1/6B1/4P3/2NQ4/PPP2PPP/3R1RK1 w - - bm e5",
    18:"q4rk1/1n1Qbppp/2p5/1p2p3/1P2P3/2P4P/6P1/2B1NRK1 b - - bm Qc8",
    19:"r2q1r1k/1b1nN2p/pp3pp1/8/Q7/PP5P/1BP2RPN/7K w - - bm Qxd7",
    20:"8/5p2/pk2p3/4P2p/2b1pP1P/P3P2B/8/7K w - - bm Bg4",
    21:"8/2k5/4p3/1nb2p2/2K5/8/6B1/8 w - - bm Kxb5",
    22:"1B1b4/7K/1p6/1k6/8/8/8/8 w - - bm Ba7",
    23:"rn1q1rk1/1b2bppp/1pn1p3/p2pP3/3P4/P2BBN1P/1P1N1PP1/R2Q1RK1 b - - bm Ba6",
    24:"8/p1ppk1p1/2n2p2/8/4B3/2P1KPP1/1P5P/8 w - - bm Bxc6",
    25:"8/3nk3/3pp3/1B6/8/3PPP2/4K3/8 w - - bm Bxd7"}
