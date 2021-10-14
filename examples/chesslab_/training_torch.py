
import torch
import torch.nn as nn
import sys
import pickle
from torch.utils.data import Dataset, DataLoader

import datetime as dt
import time
import numpy as np

from .utils import default_parameters as params

from sklearn.model_selection import train_test_split





class training:
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
            model,optimizer,loss_fn,start,encoding,history=training.load_model(load_model,model,training=True)
        else:
            optimizer=optim(model.parameters(),lr=lr,momentum=mo)


        start+=1
        NUM_EPOCHS = start+epochs

        

        for epoch in range(start,NUM_EPOCHS):
            print(f'epoch {epoch}')
            print(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            start_time = time.time()

            #Training loop
            model.train()
            loss_sum = 0
            acc_sum = 0


            percent = len_train_loader//1000
            if percent==0:
                print("Error en percent")
            for i,(x,y) in enumerate(train_loader):
                
                if i%percent == 0:
                    sys.stdout.write('\r train: {:.1f}/100 - loss:{:.4f} - acc:{:.4f}           '.format((i+1)*100/len_train_loader,loss_sum/(i+1),acc_sum/(i+1)))
                    sys.stdout.flush()

                #x=training.recode(x,inter_encoding).float().to(device)
                x=x.to(device)
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

            sys.stdout.write('\r train: {:.1f}/100 - loss:{:.4f} - acc:{:.4f}           '.format(100,loss_sum/(i+1),acc_sum/(i+1)))
            sys.stdout.flush()
            epoch_train_loss = loss_sum / len_train_loader
            epoch_train_acc = acc_sum / len_train_loader
            #print(f'valor de i:{i} i*batch={i*batch_size}')



            history["train"]["loss"].append(epoch_train_loss)
            history["train"]["acc"].append(epoch_train_acc)
            sys.stdout.write(f'\r\n')
            sys.stdout.flush()
            
            percent = len_test_loader//100
            if percent==0:
                print("Error en percent")

            if test_loader is not None:
                model.eval()

                with torch.no_grad():
                    loss_sum = 0
                    acc_sum = 0


                    for i,(x,y) in enumerate(test_loader):

                        if i%percent == 0:
                            sys.stdout.write('\r test: {:.0f}/100 - loss:{:.4f} - acc:{:.4f}           '.format((i+1)*100/len_test_loader,loss_sum/(i+1),acc_sum/(i+1)))
                            sys.stdout.flush()

                        #x=training.recode(x,inter_encoding).float().to(device)
                        x=x.to(device)
                        y=y.long().to(device)
                        #Forward
                        out = model(x)

                        loss=loss_fn(out,y[:,1])

                        #predicted_class = torch.round(out)
                        predicted_class = torch.argmax(out, axis=-1)
                        acc = torch.eq(predicted_class, y[:,1]).float().mean()#.item()


                        loss_sum += abs(loss.item())
                        acc_sum += acc.item()

                    sys.stdout.write('\r test: {:.0f}/100 - loss:{:.4f} - acc:{:.4f}           '.format(100,loss_sum/(i+1),acc_sum/(i+1)))
                    sys.stdout.flush()
                    epoch_test_loss = loss_sum / len_test_loader
                    epoch_test_acc = acc_sum / len_test_loader
                    #print(f'valor de i:{i} i*batch={i*batch_size}')



                    history["test"]["loss"].append(epoch_test_loss)
                    history["test"]["acc"].append(epoch_test_acc)



            elapsed_time = time.time() - start_time

            name=f'{save_name}.{epoch}'
            #scripted_model = torch.jit.script(model())
            #torch.jit.save(scripted_model,name+'_model.pt')
            torch.save({
                'epoch': epoch,
                #'model': traced_model,
                'model_state_dict': model.state_dict(),
                'loss_fn': loss_fn,
                'optim':optim,
                'optimizer_state_dict': optimizer.state_dict(),
                'history':history,
                'torch_rng_state':torch.get_rng_state(),
                'numpy_rng_state':np.random.get_state(),
                'encoding':encoding
                }, name+'.pt')

            
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

    inter_encoding_0={
        0:np.array([0,0,0,0],dtype=np.int8),
        1:np.array([1,0,0,0],dtype=np.int8),
        2:np.array([0,0,0,1],dtype=np.int8),
        3:np.array([0,1,0,0],dtype=np.int8),
        4:np.array([0,0,1,0],dtype=np.int8),
        5:np.array([1,1,0,0],dtype=np.int8),
        6:np.array([0,0,1,1],dtype=np.int8),
        7:np.array([1,0,1,0],dtype=np.int8),
        8:np.array([0,1,0,1],dtype=np.int8),
        9:np.array([1,0,0,1],dtype=np.int8),
        10:np.array([0,1,1,0],dtype=np.int8),
        11:np.array([1,1,1,0],dtype=np.int8),
        12:np.array([0,1,1,1],dtype=np.int8)
    }

    inter_encoding_1={
        0:torch.tensor([0,0,0,0],dtype=torch.float),
        1:torch.tensor([1,0,0,0],dtype=torch.float),
        2:torch.tensor([0,0,0,1],dtype=torch.float),
        3:torch.tensor([0,1,0,0],dtype=torch.float),
        4:torch.tensor([0,0,1,0],dtype=torch.float),
        5:torch.tensor([1,1,0,0],dtype=torch.float),
        6:torch.tensor([0,0,1,1],dtype=torch.float),
        7:torch.tensor([1,0,1,0],dtype=torch.float),
        8:torch.tensor([0,1,0,1],dtype=torch.float),
        9:torch.tensor([1,0,0,1],dtype=torch.float),
        10:torch.tensor([0,1,1,0],dtype=torch.float),
        11:torch.tensor([1,1,1,0],dtype=torch.float),
        12:torch.tensor([0,1,1,1],dtype=torch.float)
    }

    def set_encoding(encoding):
        training.keys = torch.tensor([params.inter_map[i] for i in encoding.keys()])
        training.values = torch.stack([value for value in encoding.values()],0)


    @staticmethod
    def recode(input,keys,values):
        to_return=torch.zeros([input.shape[0],len(values[0]),64])
        for i,value in enumerate(values):
            to_change=torch.where(input==keys[i])
            to_return[to_change[0],:,to_change[1]]=value
        return to_return.view([-1,len(values[0]),8,8])

    @staticmethod
    def encode(board,encoding):
        b=str(board).replace(' ','').split('\n')
        a=torch.zeros([len(encoding['.']),8,8],dtype=torch.float)
        for i,row in enumerate(b):
            for j,val in enumerate(row):
                a[:,i,j]=encoding[val]
        return a

    #keys = torch.tensor([params.inter_map[i] for i in encoding_2.keys()])
    #values = torch.stack([value for value in encoding_2.values()],0)

    class RecodeTorch:

        def __init__(self, data, keys, values ):
            transposed_data = list(zip(*data))
            x = torch.stack(transposed_data[0], 0)
            self.y = torch.stack(transposed_data[1], 0)
            self.x = training.recode(x,keys,values)
        def pin_memory(self):
            self.x = self.x.pin_memory()
            self.y = self.y.pin_memory()
            return self.x,self.y

    
    class Collate_class_wrapper:
        def __init__(self,keys,values):
            self.keys=keys
            self.values=values
        def __call__(self,batch):
            return training.RecodeTorch(batch,self.keys,self.values)
        
        
    def collate_wrapper(batch):
        return training.RecodeTorch(batch)

    def DataLoader(x_data,y_data,batch_size,shuffle,encoding):
        dataset=training.BoardsDataset( x_data = x_data, y_data=y_data )
        keys = torch.tensor([params.inter_map[i] for i in encoding.keys()])
        values = torch.stack([value for value in encoding.values()],0)
        collate_w = training.Collate_class_wrapper(keys,values)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collate_w,
            pin_memory=True,
            num_workers=4
            )


    class BoardsDataset(Dataset):

        def __init__(self,x_data=None,y_data=None):
            self.samples=torch.tensor(x_data,requires_grad=False)
            #self.samples=training.recode(self.samples,training.keys,training.values)
            self.labels=torch.tensor(y_data,requires_grad=False)
            
        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            return self.samples[idx],self.labels[idx]

    #Architecture
    def load_model(filename,model,training=False):
        checkpoint = torch.load(filename)
        #model = checkpoint['model']
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

