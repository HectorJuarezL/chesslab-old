import datetime as dt
import time
import numpy as np
import tensorflow as tf

from .utils import load_pkl,save_pkl,params,Print_r


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(model,tdata, labels,optimizer,loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(tdata)
        loss = loss_fn(labels, predictions)
   
    gradients = tape.gradient(loss, model.trainable_variables)
    capped_grads_and_vars = [(grad,model.trainable_variables[index]) for index, grad in enumerate(gradients)]
    optimizer.apply_gradients(capped_grads_and_vars)
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model,tdata, labels,loss_fn):
    predictions = model(tdata)
    t_loss = loss_fn(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

def fitting(start=0,
            epochs=1,
            x_train=None,
            y_train=None,
            x_test=None,
            y_test=None,
            model=None,
            optimizer=None,
            batch_size=128,
            lr=0.1,
            loss_fn=None,
            save_name = 'model',
            encoding=None,
            load_name = None,
            shuffle_train=True,
            shuffle_test=False,
            download_models=False):

    history={'train':{'acc':[],'loss':[]}, 'test':{'acc':[],'loss':[]} }

    if optimizer is None:
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr)
    if loss_fn is None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if load_name is not None:
        encoding,history,start = load_model(model,load_name,training=True)

    
    train_loader=data_loader( x_data = x_train,y_data=y_train,batch_size=batch_size,shuffle=True ,encoding = encoding )
    test_loader = None
    if x_test is not None and y_test is not None:
        test_loader=data_loader( x_data = x_test,y_data=y_test,batch_size=batch_size,shuffle=False , encoding = encoding )

    len_train_loader=len(train_loader)
    len_test_loader=len(test_loader)
    percent_train = len_train_loader//100
    percent_test = len_test_loader//100

    start+=1
    NUM_EPOCHS = start+epochs

    date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(date)

    for epoch in range(start,NUM_EPOCHS):
        start_time=time.time()
        print_r = Print_r() #new pretty line
        
        for i,(batch_x, batch_y) in enumerate(train_loader):
            train_step(model,batch_x,batch_y,optimizer,loss_fn)
            if percent_train<10 or i%percent_train == 0:
                print_r('Epoch: {:02}/{:02} | train progress: {:.0f}/100 | train loss:{:.4f} | train acc: {:.4f}           '
                    .format(epoch,NUM_EPOCHS-1,(i+1)*100/len_train_loader,train_loss.result(),train_accuracy.result()))
        
        if test_loader is not None:
            for i,(batch_x, batch_y) in enumerate(test_loader):
                test_step(model,batch_x,batch_y,loss_fn)
                if percent_test<10 or i%percent_test == 0:
                    print_r('Epoch: {:02}/{:02} | test progress: {:.0f}/100 | test loss:{:.4f} | test acc: {:.4f}           '
                        .format(epoch,NUM_EPOCHS-1,(i+1)*100/len_test_loader,test_loss.result(),test_accuracy.result()))


        #template = 'Epoch {:02d}{}, Tiempo: {:.1f}m Perdida: {:.6f}, Exactitud: {:.2f}, Perdida de prueba: {:.6f}, Exactitud de prueba: {:.2f}'
        history['train']['acc']=train_accuracy.result()
        history['train']['loss']=train_loss.result()
        history['test']['acc']=test_accuracy.result()
        history['test']['loss']=test_loss.result()

        elapsed_time = time.time() - start_time
        name="{}.{}.h5".format(save_name,epoch)
        save_model(model=model,history=history,encoding=encoding,epoch=epoch,name=name)

        if download_models:
            from google.colab import files
            files.download(name)

        if test_loader is not None:
            print_r('Epoch: {:02}/{:02} | time: {:.0f}s = {:.1f}m | train loss: {:.4f} | train acc: {:.4f} | test loss: {:.4f} | test acc: {:.4f}'
                .format(epoch,NUM_EPOCHS-1,elapsed_time,elapsed_time/60,train_loss.result(),train_accuracy.result(),test_loss.result(),test_accuracy.result()))
        else:
            print_r('Epoch: {:02}/{:02} | time: {:.0f}s = {:.1f}m | train loss: {:.4f} | train acc: {:.4f}'
                .format(epoch,NUM_EPOCHS-1,elapsed_time,elapsed_time/60,train_loss.result(),train_accuracy.result()))



        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

def encode(board,encoding):
    b=str(board).replace(' ','').split('\n')
    a=np.zeros([8,8,len(encoding['.'])])
    for i,row in enumerate(b):
        for j,val in enumerate(row):
            a[i,j,:]=encoding[val]
    return a

def save_model(model,history,encoding,epoch,name):
    save_pkl((model.hw,model.hb,history,encoding,epoch),name)
        
def load_model(model,name,training=False):
    (model.hw,model.hb,history,encoding,epoch)=load_pkl(name)
    if training:
        model.trainable_variables = []
        for i in range(len(model.hw)):
            model.trainable_variables.append(model.hw[i])    
            model.trainable_variables.append(model.hb[i])
        return encoding,history,epoch
    else:
        return encoding,history


def data_loader(x_data,y_data,batch_size,shuffle=True,encoding = None, seed=0):
    class Wrapper():
        def __init__(self,encoding):
            self.keys = np.array([params.inter_map[i] for i in encoding.keys()],dtype=np.int8)
            self.values = np.stack([value for value in encoding.values()],0).astype(np.float32)
        def __call__(self,x_in,y_in):
            x_in_encoded = tf.numpy_function(
                Wrapper.recode,
                inp=(x_in,self.keys,self.values),
                Tout=(tf.float32)
            )
            return x_in_encoded,y_in
            
        @staticmethod
        def recode(x_in,keys,values):
            to_return=np.zeros([x_in.shape[0],64,len(values[0])],dtype=np.float32)
            for i,value in enumerate(values):
                to_change=np.where(x_in==keys[i])
                to_return[to_change[0],to_change[1],:]=value
            return np.reshape(to_return,(-1,8,8,len(values[0])))

    
    x_data=tf.data.Dataset.from_tensor_slices(x_data)
    y_data=tf.data.Dataset.from_tensor_slices(y_data.astype(np.float32))
    
    dataset=tf.data.Dataset.zip((x_data,y_data))
    dataset = dataset.shuffle(buffer_size=len(dataset),seed=seed,reshuffle_each_iteration=shuffle)
    dataset = dataset.batch(batch_size)
    if encoding is not None:
        wrapper = Wrapper(encoding)
        dataset=dataset.map(wrapper,num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

