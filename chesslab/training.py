
import torch
import torch.nn as nn
import sys
import pickle
from torch.utils.data import Dataset, DataLoader

import datetime as dt
import time
import numpy as np

from chesslab.base import load_pkl
from chesslab.base import default_parameters as params


from sklearn.model_selection import train_test_split


class torch_architectures:
    lr = 0.1
    mo = 0.1

    loss_fn=nn.CrossEntropyLoss(reduction='mean')
    optim=torch.optim.SGD


    def train(start=0,epochs=1,train_loader=None,test_loader=None,
        device=None,model=None,optim=None,lr=lr,mo=mo,loss_fn=None,
        save_name='model',encoding=None,load_model=None):

        len_train_loader=len(train_loader)
        history = {"train": {"loss": [], "acc": []}}

        if test_loader is not None:
            history = {"train": {"loss": [], "acc": []},"test": {"loss": [], "acc": []}}
            len_test_loader=len(test_loader)

        if load_model is not None:
            model,optimizer,loss_fn,start,encoding,history=torch_architectures.load_model(load_model,training=True)
        else:
            optimizer=optim(model.parameters(),lr=lr,momentum=mo)

        start+=1
        NUM_EPOCHS = start+epochs

        inter_encoding = {params.inter_map[i]:code for i,code in encoding.items()}



        for epoch in range(start,NUM_EPOCHS):
            print(f'epoch {epoch}')
            print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            start_time = time.time()

            #Training loop
            model.train()
            loss_sum = 0
            acc_sum = 0



            for i,(x,y) in enumerate(train_loader):
                
                sys.stdout.write(f'\r train: {(i+1)*100/len_train_loader:.2f}/100 - loss:{loss_sum/(i+1):.4f} - acc:{acc_sum/(i+1):.4f}           ')
                sys.stdout.flush()

                x=torch_architectures.recode(x,inter_encoding).float().to(device)
                y=y.long().to(device)
                #Forward
                out = model(x)
                #>>logits: #######


                #Backprop
                optimizer.zero_grad()

                loss=loss_fn(out,y[:,1])

                #predicted_class = torch.round(out)
                predicted_class = torch.argmax(out, axis=-1)
                acc = torch.eq(predicted_class, y[:,1]).float().mean()#.item()

                loss.backward()

                #Update params
                optimizer.step()

                loss_sum += abs(loss.item())
                acc_sum += acc.item()


            epoch_train_loss = loss_sum / len_train_loader
            epoch_train_acc = acc_sum / len_train_loader
            #print(f'valor de i:{i} i*batch={i*batch_size}')



            history["train"]["loss"].append(epoch_train_loss)
            history["train"]["acc"].append(epoch_train_acc)
            sys.stdout.write(f'\r\n')
            sys.stdout.flush()

            if test_loader is not None:
                model.eval()

                with torch.no_grad():
                    loss_sum = 0
                    acc_sum = 0


                    for i,(x,y) in enumerate(test_loader):

                        sys.stdout.write(f'\r test: {(i+1)*100/len_test_loader:.2f}/100 - loss:{loss_sum/(i+1):.4f} - acc:{acc_sum/(i+1):.4f}           ')
                        sys.stdout.flush()

                        x=torch_architectures.recode(x,inter_encoding).float().to(device)
                        y=y.long().to(device)
                        #Forward
                        out = model(x)
                        #>>logits: #######


                        loss=loss_fn(out,y[:,1])

                        #predicted_class = torch.round(out)
                        predicted_class = torch.argmax(out, axis=-1)
                        acc = torch.eq(predicted_class, y[:,1]).float().mean()#.item()


                        loss_sum += abs(loss.item())
                        acc_sum += acc.item()


                    epoch_test_loss = loss_sum / len_test_loader
                    epoch_test_acc = acc_sum / len_test_loader
                    #print(f'valor de i:{i} i*batch={i*batch_size}')



                    history["test"]["loss"].append(epoch_test_loss)
                    history["test"]["acc"].append(epoch_test_acc)



            elapsed_time = time.time() - start_time

            NAME=f'{save_name}.{epoch}.h5'
            torch.save({
                'epoch': epoch,
                'model': model,
                'model_state_dict': model.state_dict(),
                'loss_fn': loss_fn,
                'optim':optim,
                'optimizer_state_dict': optimizer.state_dict(),
                'history':history,
                'torch_rng_state':torch.get_rng_state(),
                'numpy_rng_state':np.random.get_state(),
                'encoding':encoding
                }, NAME)

            
            if test_loader is not None:
                print(f'\nEpoch: {epoch:03}/{NUM_EPOCHS-1} | Time: {elapsed_time:.0f}s = {elapsed_time/60:.1f}m | Train loss: {epoch_train_loss:.4f} | Train acc: {epoch_train_acc:.4f} | Test loss: {epoch_test_loss:.4f} | Test acc: {epoch_test_acc:.4f}')
            else:
                print(f'\nEpoch: {epoch:03}/{NUM_EPOCHS-1} | Time: {elapsed_time:.0f}s = {elapsed_time/60:.1f}m | Train loss: {epoch_train_loss:.4f} | Train acc: {epoch_train_acc:.4f}')
            print('\n'+'-' * 80)


    encoding_1={
        '.':torch.tensor([0,0,0],dtype=torch.float),
        'p':torch.tensor([0,0,1],dtype=torch.float),
        'P':torch.tensor([0,0,-1],dtype=torch.float),
        'b':torch.tensor([0,1,0],dtype=torch.float),
        'B':torch.tensor([0,-1,0],dtype=torch.float),
        'n':torch.tensor([1,0,0],dtype=torch.float),
        'N':torch.tensor([-1,0,0],dtype=torch.float),
        'r':torch.tensor([0,1,1],dtype=torch.float),
        'R':torch.tensor([0,-1,-1],dtype=torch.float),
        'q':torch.tensor([1,0,1],dtype=torch.float),
        'Q':torch.tensor([-1,0,-1],dtype=torch.float),
        'k':torch.tensor([1,1,0],dtype=torch.float),
        'K':torch.tensor([-1,-1,0],dtype=torch.float)
    }

    encoding_2={
        '.':torch.tensor([0,0,0,0],dtype=torch.float),
        'p':torch.tensor([1,0,0,0],dtype=torch.float),
        'P':torch.tensor([0,0,0,1],dtype=torch.float),
        'b':torch.tensor([0,1,0,0],dtype=torch.float),
        'B':torch.tensor([0,0,1,0],dtype=torch.float),
        'n':torch.tensor([1,1,0,0],dtype=torch.float),
        'N':torch.tensor([0,0,1,1],dtype=torch.float),
        'r':torch.tensor([1,0,1,0],dtype=torch.float),
        'R':torch.tensor([0,1,0,1],dtype=torch.float),
        'q':torch.tensor([1,0,0,1],dtype=torch.float),
        'Q':torch.tensor([0,1,1,0],dtype=torch.float),
        'k':torch.tensor([1,1,1,0],dtype=torch.float),
        'K':torch.tensor([0,1,1,1],dtype=torch.float)
    }

    def recode(x_in,encoding):
        to_return=torch.zeros([x_in.shape[0],len(encoding[0]),64])
        for key,value in encoding.items():
            to_change=torch.where(x_in==key)
            to_return[to_change[0],:,to_change[1]]=value
        return to_return.view([-1,len(encoding[0]),8,8])

    def encode(board,encoding):
        b=str(board).replace(' ','').split('\n')
        a=torch.zeros([len(encoding['.']),8,8],dtype=torch.float)
        for i,row in enumerate(b):
            for j,val in enumerate(row):
                a[:,i,j]=encoding[val]
        return a

    def DataLoader(x_data,y_data,batch_size,shuffle):
        dataset=torch_architectures.BoardsDataset( x_data = x_data,y_data=y_data)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    class BoardsDataset(Dataset):

        def __init__(self,x_data=None,y_data=None):
            self.samples=x_data
            self.labels=y_data
            
        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            
            return self.samples[idx],self.labels[idx]

    #Architecture
    def load_model(filename,training=False):
        checkpoint = torch.load(filename)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        encoding=checkpoint['encoding']
        if training:
            optim=checkpoint['optim']
            optimizer = optim(model.parameters(),lr=0.2)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss_fn=checkpoint['loss_fn']
            start = checkpoint['epoch']
            np.random.set_state(checkpoint['numpy_rng_state'])
            torch.set_rng_state(checkpoint['torch_rng_state'])
            return model,optimizer,loss_fn,start,encoding,checkpoint['history']
        else:
            return model,encoding,checkpoint['history']


    class Model_1(nn.Module):
        
        def __init__(self):
            super().__init__()
            # Encoder specification
            self.cnn_1 = nn.Conv2d(3, 64, kernel_size=7,padding=3)
            self.func_1=nn.ELU()
            self.cnn_2 = nn.Conv2d(64, 128, kernel_size=5,padding=2)
            self.func_2=nn.ELU()
            self.cnn_3 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
            self.func_3=nn.ELU()
            self.cnn_4 = nn.Conv2d(256, 64, kernel_size=1,padding=0)
            self.func_4=nn.ELU()
            self.cnn_5 = nn.Conv2d(64, 16, kernel_size=1,padding=0)
            self.func_5=nn.ELU()
            self.cnn_6 = nn.Conv2d(16, 4, kernel_size=1,padding=0)
            self.func_6=nn.ELU()
            self.linear_1 = nn.Linear(8*8*4,64 )
            self.func_7=nn.ELU()
            self.linear_2 = nn.Linear(64, 2)
            
        def forward(self, x):
            out = self.func_1(self.cnn_1(x))
            out = self.func_2(self.cnn_2(out))
            out = self.func_3(self.cnn_3(out))
            out = self.func_4(self.cnn_4(out))
            out = self.func_5(self.cnn_5(out))
            out = self.func_6(self.cnn_6(out))
            out = out.reshape([x.size(0), -1])
            out = self.func_7(self.linear_1(out))
            out = self.linear_2(out)
            
            return out
    class Model_2(nn.Module):
    
        def __init__(self):
            super().__init__()
            # Encoder specification
            self.cnn_1 = nn.Conv2d(4, 64, kernel_size=7,padding=3)
            self.func_1=nn.ELU()
            self.cnn_2 = nn.Conv2d(64, 128, kernel_size=5,padding=2)
            self.func_2=nn.ELU()
            self.cnn_3 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
            self.func_3=nn.ELU()
            self.cnn_4 = nn.Conv2d(256, 16, kernel_size=1,padding=0)
            self.func_4=nn.ELU()
            self.cnn_5 = nn.Conv2d(16, 1, kernel_size=1,padding=0)
            self.func_5=nn.ELU()
            self.linear_1 = nn.Linear(8*8,32 )
            self.func_7=nn.ELU()
            self.linear_2 = nn.Linear(32, 2)
            
        def forward(self, x):
            out = self.func_1(self.cnn_1(x))
            out = self.func_2(self.cnn_2(out))
            out = self.func_3(self.cnn_3(out))
            out = self.func_4(self.cnn_4(out))
            out = self.func_5(self.cnn_5(out))
            out = out.reshape([x.size(0), -1])
            out = self.func_7(self.linear_1(out))
            out = self.linear_2(out)
            
            return out



if __name__ == '__main__':
    np.random.seed(0)

    batch_size = params.batch_size
    #path='../new_encode/'
    path = 'D:/Database_encoded/ccrl/'
    #path='E:/Archivos/new_pgn/'



    name=f'{path}ccrl_states_c.pkl'
    x_data = load_pkl(name)[:100000]

    name=f'{path}ccrl_results_c.pkl'
    y_data = load_pkl(name)[:100000]

    encoding=torch_architectures.encoding_1

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size= params.test_size, random_state=0,shuffle=False)

    del x_data
    del y_data


    train_loader=torch_architectures.DataLoader(path, x_data = x_train,y_data=y_train,batch_size=batch_size,shuffle=True)

    test_loader=torch_architectures.DataLoader(path, x_data = x_test,y_data=y_test,batch_size=batch_size,shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_name='noEat_0.2'

    lr = torch_architectures.lr
    mo = torch_architectures.mo
    start=0
    # Instantiate model


    model = torch_architectures.Model_1().to(device)
    loss_fn = torch_architectures.loss_fn
    optim = torch_architectures.optim



    print(model)
    torch_architectures.train(start=1,epochs=2,train_loader=train_loader,test_loader=test_loader,
             device=device,model=model,optim=optim,lr=lr,mo=mo,
             loss_fn=loss_fn,save_name=save_name,encoding=encoding)






