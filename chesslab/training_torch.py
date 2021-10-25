import datetime as dt
import time
import numpy as np
import torch
import os

from .utils import params,Print_r

def fitting(start=0,
            epochs=1,
            x_train=None,
            y_train=None,
            x_test=None,
            y_test=None,
            device=None,
            model=None,
            optim=None,
            batch_size=128,
            lr=0.1,
            mo=0,
            loss_fn=None,
            save_name='model',
            encoding=None,
            load_name=None,
            shuffle_train=True,
            shuffle_test=False,
            num_workers=0,
            download_models=False):


    
    history = {"train": {"loss": [], "acc": []},"test": {"loss": [], "acc": []}}
    if load_name is not None:
        model,optimizer,loss_fn,start,encoding,history=load_model(model,load_name,training=True,device=device)
    else:
        optimizer=optim(model.parameters(),lr=lr,momentum=mo)

    train_loader = data_loader( x_data = x_train, y_data=y_train,batch_size=batch_size, shuffle=shuffle_train,encoding = encoding, num_workers=num_workers )
    test_loader = None
    if x_test is not None and y_test is not None:
        test_loader=data_loader( x_data = x_test, y_data=y_test, batch_size=batch_size, shuffle=shuffle_test, encoding = encoding, num_workers=num_workers )
        len_test_loader=len(test_loader)

    len_train_loader=len(train_loader)

    start+=1
    NUM_EPOCHS = start+epochs

    folder = os.path.dirname(os.path.abspath(save_name))
    if not os.path.exists(folder):
        os.makedirs(folder)

    percent_train = len_train_loader//100
    percent_test = len_test_loader//100
    date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(date)

    for epoch in range(start,NUM_EPOCHS):
        
        start_time = time.time()

        print_r = Print_r()
        model.train()
        loss_sum = 0
        acc_sum = 0


        for i,(x,y) in enumerate(train_loader):
            if percent_train<10 or i%percent_train == 0:
                print_r('Epoch: {:02}/{:02} | train progress: {:.0f}/100 | train loss: {:.4f} | train acc: {:.4f}'.format(epoch,NUM_EPOCHS-1,i/len_train_loader*100,loss_sum/(i+1),acc_sum/(i+1)))
            x=x.to(device)
            y=y.long().to(device)
            
            out = model(x)

            optimizer.zero_grad()
            loss=loss_fn(out,y) 
            predicted_class = torch.argmax(out, axis=-1)
            acc = torch.eq(predicted_class, y).float().mean()#.item()
            loss.backward()
            optimizer.step()

            loss_sum += abs(loss.item())
            acc_sum += acc.item()

        epoch_train_loss = loss_sum / len_train_loader
        epoch_train_acc = acc_sum / len_train_loader

        history["train"]["loss"].append(epoch_train_loss)
        history["train"]["acc"].append(epoch_train_acc)

        if test_loader is not None:
            model.eval()

            with torch.no_grad():
                loss_sum = 0
                acc_sum = 0
                for i,(x,y) in enumerate(test_loader):
                    if percent_test<10 or i%percent_test == 0:
                        print_r('Epoch: {:02}/{:02} | test progress: {:.0f}/100 | test loss: {:.4f} | test acc: {:.4f}'.format(epoch,NUM_EPOCHS-1,i/len_test_loader*100,loss_sum/(i+1),acc_sum/(i+1)))
                    x=x.to(device)
                    y=y.long().to(device)
                    out = model(x)
                    loss=loss_fn(out,y) 
                    predicted_class = torch.argmax(out, axis=-1)
                    acc = torch.eq(predicted_class, y).float().mean()#.item()
                    loss_sum += abs(loss.item())
                    acc_sum += acc.item()

                epoch_test_loss = loss_sum / len_test_loader
                epoch_test_acc = acc_sum / len_test_loader

                history["test"]["loss"].append(epoch_test_loss)
                history["test"]["acc"].append(epoch_test_acc)

        elapsed_time = time.time() - start_time

        name=f'{save_name}.{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn': loss_fn,
            'optim':optim,
            'optimizer_state_dict': optimizer.state_dict(),
            'history':history,
            'torch_rng_state':torch.get_rng_state(),
            'numpy_rng_state':np.random.get_state(),
            'lr':lr,
            'encoding':encoding
            }, name)

        if download_models:
            from google.colab import files
            files.download(name)

        if test_loader is not None:
            print_r('Epoch: {:02}/{:02} | time: {:.0f}s = {:.1f}m | train loss: {:.4f} | train acc: {:.4f} | test loss: {:.4f} | test acc: {:.4f}'
                .format(epoch,NUM_EPOCHS-1,elapsed_time,elapsed_time/60,epoch_train_loss,epoch_train_acc,epoch_test_loss,epoch_test_acc))
        else:
            print_r('Epoch: {:02}/{:02} | time: {:.0f}s = {:.1f}m | train loss: {:.4f} | train acc: {:.4f}'
                .format(epoch,NUM_EPOCHS-1,elapsed_time,elapsed_time/60,epoch_train_loss,epoch_train_acc))

def encode(board,encoding):
    b=str(board).replace(' ','').split('\n')
    a=torch.zeros([len(encoding['.']),8,8],dtype=torch.float)
    for i,row in enumerate(b):
        for j,val in enumerate(row):
            a[:,i,j]=encoding[val]
    return a

def load_model(model,filename,training=False,device=None):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    encoding=checkpoint['encoding']
    if training:
        optim=checkpoint['optim']
        optimizer = optim(model.parameters(),lr=checkpoint['lr'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_fn=checkpoint['loss_fn']
        start = checkpoint['epoch']
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        return model,optimizer,loss_fn,start,encoding,checkpoint['history']
    else:
        return model,encoding,checkpoint['history']

class RecodeTorch:
    def __init__(self, data, keys, values ):
        transposed_data = list(zip(*data))
        x = torch.stack(transposed_data[0], 0)
        self.y = torch.stack(transposed_data[1], 0)
        self.x = torch.zeros([x.shape[0],len(values[0]),64])
        for i,value in enumerate(values):
            to_change=torch.where(x==keys[i])
            self.x[to_change[0],:,to_change[1]]=value
        self.x = self.x.view([-1,len(values[0]),8,8])
    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.y = self.y.pin_memory()
        return self.x,self.y


class Collate_class_wrapper:
    import torch
    def __init__(self,encoding):
        self.keys = torch.tensor([params.inter_map[i] for i in encoding.keys()])
        self.values = torch.stack([value for value in encoding.values()],0)
    def __call__(self,batch):
        return RecodeTorch(batch,self.keys,self.values)



class BoardsDataset(torch.utils.data.Dataset):
    def __init__(self,x_data=None,y_data=None):
        self.samples=torch.tensor(x_data,requires_grad=False)
        self.labels=torch.tensor(y_data,requires_grad=False)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.samples[idx],self.labels[idx]

def data_loader(x_data,y_data,batch_size,shuffle=True,encoding=None,seed=0,num_workers=0):
    dataset=BoardsDataset( x_data = x_data, y_data=y_data )
    collate_w = Collate_class_wrapper(encoding) if encoding is not None else None
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_w,
        pin_memory=True,
        num_workers=num_workers
        )

