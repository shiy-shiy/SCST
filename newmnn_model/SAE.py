#one center of sc+st
#add BN after encoder_1 + regularizers
#add BN after encoder_1
#BN 2FIT
#fit(y=autoencoder隐层输出) y_true是512*32的信息阵：self.encoder.predict(x)
#loss构造：sc/st的信息阵做概率，两个独立的分布，用交叉熵可以使两分布相似
import os
os.environ['PYTHONHASHSEED'] = '0'
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
import  math
import numpy as np
import random
import tensorflow as tf
from random import sample
import scanpy as sc
import pandas as pd
#random.seed(201809)
#np.random.seed(201809)
#tf.set_random_seed(201809) if tf.__version__<="2.0" else tf.random.set_seed(201809)
import math
import time
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



class SAE(object):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    
    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)
        
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model.
        random_seed, `int`,optional. Default,`201809`. the random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed
        actincenter: the activation function in last layer for encoder and last layer for encoder (avoiding the representation values and reconstruct outputs are all non-negative)
        init: `str`,optional. Default: `glorot_uniform`. Initialization method used to initialize weights.
        use_earlyStop: optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.
        save_dir:'str',optional. Default,'result_tmp',some result will be saved in this directory.
    """
    
    def __init__(self, dims, obs,x,act='relu', 
            drop_rate=0.2, 
            batch_size=1024, #512/8961
            random_seed=201809,
            actincenter="tanh",
            init="glorot_uniform",
            use_earlyStop=True,
            save_dir='result_tmp'): #act relu

        
        self.dims = dims
        self.obs=obs
        self.x=x
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        self.actincenter=actincenter #linear
        self.drop_rate = drop_rate
        self.init=init
        self.batch_size = batch_size
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        #tf.set_random_seed(random_seed)
        tf.set_random_seed(random_seed) if tf.__version__<"2.0" else tf.random.set_seed(random_seed)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #
        self.random_seed=random_seed
        self.use_earlyStop=use_earlyStop
        self.stacks = [self.make_stack(i,random_seed=self.random_seed+2*i) for i in range(self.n_stacks)]
        self.firstfit,self.autoencoders ,self.encoder,self.finetune= self.make_autoencoders()
        #plot_model(self.autoencoders, show_shapes=True, to_file=os.path.join(save_dir,'autoencoders.png'))        
        #idx=st_sample.index
 


    def choose_init(self,init="glorot_uniform",seed=1):
        if init not in {'glorot_uniform','glorot_normal','he_normal','lecun_normal','he_uniform','lecun_uniform','RandomNormal','RandomUniform',"TruncatedNormal"}:
            raise ValueError('Invalid `init` argument: '
                             'expected on of {"glorot_uniform", "glorot_normal", "he_normal","he_uniform","lecun_normal","lecun_uniform","RandomNormal","RandomUniform","TruncatedNormal"} '
                             'but got', mode)
        """
        #tensorflow <2.0
        if init=="glorot_uniform":
            res=keras.initializers.glorot_uniform(seed=seed)
        elif init=="glorot_normal":
            res=keras.initializers.glorot_normal(seed=seed)
        elif init=="he_normal":
            res=keras.initializers.he_normal(seed=seed)
        elif init=='he_uniform':
            res=keras.initializers.he_uniform(seed=seed)
        elif init=="lecun_normal":
            res=keras.initializer.lecun_normal(seed=seed)
        elif init=="lecun_uniform":
            res=keras.initializers.lecun_uniform(seed=seed)
        elif init=="RandomNormal":
            res=keras.initializers.RandomNormal(mean=0.0,stddev=0.04,seed=seed)
        elif init=="RandomUniform":
            res=keras.initializers.RandomUniform(minval=-0.05,maxval=0.05,seed=seed)
        else:
            res=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed)
        """
        return init
        
        

    def make_autoencoders(self):
        """ Fully connected autoencoders model, symmetric.
        """
        # input
        # input=sc+(2*sc)st
        
        x = Input(shape=(self.dims[0],), name='input')  #inputlayer
        print('x',x)
        h = x
        
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+i),activation=self.activation, name='encoder_%d' % i)(h)
            
        h = Dense(self.dims[-1],kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks), name='encoder_%d' % (self.n_stacks - 1),activation=self.actincenter)(h)  # features are extracted from here

        h = tf.keras.layers.BatchNormalization(name='BatchNormalization_2')(h)
        h=tf.keras.layers.Dropout(0.3)(h)
        t = Dense(len(self.obs['desc_0.2'].cat.categories), kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks), name='fine-tuning',activation="softmax")(h)
#len(self.obs['annotation_1'].cat.categories)
        y=h
        # internal layers in decoder       
        for i in range(self.n_stacks-1, 0, -1):
            y = Dense(self.dims[i], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks+i),activation=self.activation, name='decoder_%d' % i)(y)
        
        # output
        y = Dense(self.dims[0], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+2*self.n_stacks),name='decoder_0',activation="linear")(y)

        return Model(inputs=x, outputs=y,name="auto"),Model(inputs=x, outputs=[y,h],name="AE"),Model(inputs=x,outputs=h,name="encoder"),    Model(inputs=x,outputs=[y,t],name='finetune')   #Model(inputs=x,outputs=t,name='finetune')#,BatchNormalization_0,BatchNormalization_1,BatchNormalization_2,BatchNormalization_3    #AEoutput=y,encoderoutput=h
        #crossentropy
        #self.firstfit,self.autoencoders ,self.encoder,self.finetune= self.make_autoencoders()
        #exit()

        

    def make_stack(self, ith,random_seed=0):
        """ 
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is 
        corrupted by Dropout(drop_rate)
        
        Arguments:
            ith: int, in [0, self.n_stacks)
        """
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        output_act = self.activation
        hidden_act = self.activation
        if ith == 0:
            output_act = self.actincenter #tanh, or linear
        if ith == self.n_stacks-1:
            hidden_act = self.actincenter #tanh, or linear
        model = Sequential()
        model.add(Dropout(self.drop_rate, input_shape=(in_out_dim,),seed=random_seed))
        model.add(Dense(units=hidden_dim, activation=hidden_act, kernel_initializer=self.choose_init(init=self.init,seed=random_seed),name='encoder_%d' % ith))
        model.add(Dropout(self.drop_rate,seed=random_seed+1))
        model.add(Dense(units=in_out_dim, activation=output_act,kernel_initializer=self.choose_init(init=self.init,seed=random_seed+1), name='decoder_%d' % ith))
        return model

    def pretrain_stacks(self, x, epochs=200,decaying_step=3):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
            decayiing_step: learning rate multiplies 0.1 every 'epochs/decaying_step' epochs 
        """
        features = x
        for i in range(self.n_stacks):
            print( 'Pretraining the %dth layer...' % (i+1))
            for j in range(int(decaying_step)):  # learning rate multiplies 0.1 every 'epochs/decaying_step' epochs 向上取整
                print ('learning rate =', pow(10, -1-j))
                self.stacks[i].compile(optimizer=SGD(pow(10, -1-j), momentum=0.9), loss='mse')  #stack.compile
                if self.use_earlyStop is True:
                    callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=5,verbose=1,mode='auto')]   #
                    self.stacks[i].fit(features,features,callbacks=callbacks,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step)) #stack.fit
                else:
                    self.stacks[i].fit(x=features,y=features,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
            print ('The %dth layer has been pretrained.' % (i+1))

            # update features to the inputs of the next layer
            feature_model = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features = feature_model.predict(features)

    def pretrain_autoencoders(self, x,epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        Arguments:
        x: input data, shape=(n_samples, n_dims)
        epochs: training epochs
        """
        print ('Copying layer-wise pretrained weights to deep autoencoders')
        for i in range(self.n_stacks):
            name = 'encoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name = 'decoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
        print ('Fine-tuning autoencoder end-to-end')

        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j) #10^(-j)
            '''
        initial_learning_rate = 1.0
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)'''
            print ('learning rate =', lr)
            #self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss=['mse',myloss],loss_weights=[1,1])  #,loss_weights=[1,1] self.myloss#encoder.compile 'mse' #penalized_loss(entropy)  self.penalized_loss(entropy)
            self.autoencoders.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss={'decoder_0':'categorical_crossentropy','fine-tuning':'categorical_crossentropy'},loss_weights=[1.0,1.0],metrics = {'decoder_0':'accuracy','fine-tuning':'categorical_accuracy'})
            print("mse loss done")
            callbacks=[EarlyStopping(monitor='loss',min_delta=1e-3,patience=10,verbose=1,mode='auto')]
            #checkpoint_filepath = '/s/f/shiyi/ST/hierarchy_dense_net/test1029/mynetwork/mynetwork_scstmerge/oldmodel/oldmodel0302/model1/weights_{epoch:03d}-{val_loss:.4f}.h5'
            #model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_weights_only=False,monitor='loss',mode='auto',save_best_only=False)
            
            self.autoencoders.fit(x=x,y=[x,self.obs['typecode']],callbacks=callbacks,batch_size=self.batch_size,epochs=50)  #y_dim  #encoder.fit#typecode
        '''
        #先对autoencoder做mse，再对encoder做entropy
        self.encoder.compile(optimizer=SGD(lr, momentum=0.9), loss=myloss) #,loss_weights=[0,1]
        print("myloss done")
        print("typecode",self.obs['typecode'])
        callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
        self.encoder.fit(x=x,y=self.obs['typecode'],callbacks=callbacks,batch_size=self.batch_size,epochs=50)#y=[x,self.obs['typecode']]
    '''
    #######

    def fit(self, x, epochs=300,decaying_step=3): # use stacked autoencoder pretrain and fine tuning
        self.pretrain_stacks(x, epochs=int(epochs/2),decaying_step=decaying_step)
        self.pretrain_autoencoders(x,epochs=epochs)
        #self.encoder
        print("fit")                                                    

    #loss怎么改？custom regularizer给loss增加了公共惩罚： ms
    # \e+entropy?

    def fit2(self,x,epochs=50): #no stack directly tran
        x_train=x[self.obs['data']=='traindata']
        X_test=x[self.obs['data']=='testdata']
        x_val=x[self.obs['data']=='validation']

        y_train=to_categorical(self.obs[self.obs['data']=='traindata']['desc_0.2'], num_classes=None)   #anno_code
        #y_test=to_categorical(self.obs['anno1_code'][self.obs['data']=='testdata'], num_classes=None)
        y_val=to_categorical(self.obs['desc_0.2'][self.obs['data']=='validation'], num_classes=None)    #anno_code
        print(x_train.shape,X_test.shape,x_val.shape)
        #print(y_train.shape,y_test.shape,y_val.shape)

        #autoencoder
        self.firstfit.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse',metrics = ['accuracy'])
        callbacks=[EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=5,verbose=1,mode='auto')]
        self.firstfit.fit(x=x_train,y=x_train,batch_size=self.batch_size,epochs=epochs,callbacks=callbacks,validation_data=(x_val,x_val))
        #reconstructure loss, fine tuning
        #self.finetune.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss={'decoder_0':'mse','fine-tuning':'categorical_crossentropy'},loss_weights=[1.0,1.0],metrics = {'decoder_0':'accuracy','fine-tuning':'categorical_accuracy'})#metrics = ['accuracy']categorical_accuracy
        #callbacks=[EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=5,verbose=1,mode='auto')]
        #self.finetune.fit(x=x_train,y=[x_train,y_train],batch_size=self.batch_size,epochs=epochs,callbacks=callbacks,validation_data=(x_val,[x_val,y_val]))#
        
        self.finetune.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss={'decoder_0':'categorical_crossentropy','fine-tuning':'categorical_crossentropy'},loss_weights=[1.0,1.0],metrics = {'decoder_0':'accuracy','fine-tuning':'categorical_accuracy'})#metrics = ['accuracy']categorical_accuracy
        callbacks=[EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=5,verbose=1,mode='auto')]
        self.finetune.fit(x=x_train,y=[x_train,y_train],batch_size=self.batch_size,epochs=epochs,callbacks=callbacks,validation_data=(x_val,[x_val,y_val]))#
        
        
        #score = self.finetune.evaluate(X_test,[X_test,y_test] , verbose=0)
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])

        
        """
        #fine tuning
        self.finetune.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',metrics = ['accuracy'])#metrics = ['accuracy']categorical_accuracy
        categorical_labels = to_categorical(self.obs['anno1_code'], num_classes=None)
        print("one-hot",categorical_labels)
        callbacks=[EarlyStopping(monitor='val_loss',min_delta=1e-2,patience=5,verbose=1,mode='auto')]
        self.finetune.fit(x=x_train,y=y_train,batch_size=self.batch_size,epochs=epochs,callbacks=callbacks,validation_data=(x_val,y_val))
        score = self.finetune.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
"""
        # make predictions
        #yhat = self.finetune.predict(X_test)
        # evaluate predictions
        #acc = accuracy_score(y_test, yhat)
        #print('Accuracy: %.3f' % acc)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders(representation).
        
        Arguments:
        x: data
        """
        X_test=x[self.obs['data']=='testdata']
        print("X_test",X_test)
        return self.finetune.predict(X_test)
        #self.finetune.predict(x)
        #return self.encoder.predict(x)
'''
def save_activations(model):
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([model.input],[out]) for out in outputs]
    layer_activations = [f([X_input_vectors]) for f in functors]
    activations_list.append(layer_activations)'''

#activations_callback = LambdaCallback(on_epoch_end = lambda batch, logs:save_activations(model))

def myloss(y_true, y_pred):
        #y_pred=self.encoder.predict(x), y_true=self.obs, self.obs['type']=='sc'=0, self.obs['type']=='st'=1
        #tf.config.run_functions_eagerly(True)
        print("y_pred",y_pred)
        print("y_true",y_true.shape)    #0-sc,1-st
        print("y_pred.unique",np.unique(y_pred.numpy(),axis=0).shape)
        #def center of scst
        #center=tf.reduce_sum(y_pred,axis=0,keepdims=True)
        #center=tf.divide(tf.reduce_sum(y_pred,axis=0,keepdims=True),y_pred.shape[0])    #sum_col/8961
        center=tf.reduce_mean(y_pred,axis=0,keepdims=True)
        print("center",center)
        center_dim=tf.tile(center,[8961,1])
        print("center_dim",center_dim)               
        #def distance
        print("sub",tf.subtract(y_pred,center_dim))
        distance=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred,center_dim)),axis=1,keepdims=True)) #euclidean distance of every point to center
        print("distance",distance)
        #dist=tf.norm(sc_colsum-st_colsum,ord='euclidean')
        #myloss=tf.divide(tf.reduce_sum(distance),y_pred.shape[0])   #sum_distance/8961
        myloss=tf.reduce_mean(distance)
        print("myloss",myloss)
        return myloss    #loss和entropy成反比
'''
def myloss(y_true, y_pred):
        #y_pred=self.encoder.predict(x), y_true=self.obs, self.obs['type']=='sc'=0, self.obs['type']=='st'=1
        #tf.config.run_functions_eagerly(True)
        print("y_pred",y_pred.shape)
        print("y_true",y_true.shape)    #0-sc,1-st
        print("y_pred.unique",np.unique(y_pred.numpy(),axis=0).shape)
        #def center of scst
        center=tf.reduce_sum(y_pred,axis=0,keepdims=True)
        print("center",center)
        #def distance
        print("sub",tf.subtract(y_pred,center))
        distance=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_pred,center)),axis=1,keepdims=True))
        print("distance",distance)
        #dist=tf.norm(sc_colsum-st_colsum,ord='euclidean')
        myloss=distance
        return myloss    #loss和entropy成反比'''


if __name__ == "__main__":
    """
    An example for how to use SAE model on MNIST dataset. You can copy this file, and run `python3 SAE.py` in terminal
    """
    import numpy as np
    def load_mnist(sample_size=10000):
        # the data, shuffled and split between train and test sets
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1))
        print ('MNIST samples', x.shape)
        id0=np.random.choice(x.shape[0],sample_size,replace=False)
        return x[id0], y[id0]

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # no use GPU
    x,y=load_mnist(10000)
    db = 'mnist'
    n_clusters = 10
    # define and train SAE model
    sae = SAE(dims=[x.shape[-1], 64,32])
    sae.fit(x=x, epochs=400)
    sae.autoencoders.save_weights('weights_%s.h5' % db)

    # extract features
    print ('Finished training, extracting features using the trained SAE model')
    features = sae.extract_feature(x)
    print ('performing k-means clustering on the extracted features')
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters, n_init=20)
    y_pred = km.fit_predict(features)
    from sklearn.metrics import normalized_mutual_info_score as nmi
    print ('K-means clustering result on extracted features: NMI =', nmi(y, y_pred))
"""
from tensorflow import keras

# 定义自编码器模型
input_shape = (784,)  # 输入数据形状为28x28的图像展平后的向量
encoding_dim = 32  # 隐层维度为32
input_data = keras.Input(shape=input_shape)
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_data)
decoded = keras.layers.Dense(input_shape[0], activation='sigmoid')(encoded)
autoencoder = keras.Model(input_data, decoded)

# 训练自编码器模型
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# 输出隐层
encoder = keras.Model(input_data, encoded)
encoded_data = encoder.predict(x_test)
print(encoded_data.shape)  # 输出隐层数据形状

"""


"""
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# split into inputs and outputs
X=
y=
print(X.shape, y.shape)

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# fit the model
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
# make predictions
yhat = model.predict(X_test)
# evaluate predictions
acc = accuracy_score(y_test, yhat)
print('Accuracy: %.3f' % acc)"""
