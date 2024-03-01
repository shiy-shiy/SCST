
"""
Keras implement Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis
"""
from __future__ import division
import os
import matplotlib
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
os.environ['PYTHONHASHSEED'] = '1'
#import networkx as nx
import matplotlib.pyplot as plt
from time import time as get_time
import numpy as np
import random
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from tensorflow.keras.layers import Dense, Input,Layer,InputSpec,Softmax
from tensorflow.keras.models import Model,load_model,Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import tensorflow as tf
import scanpy as sc
import pandas as pd
import scanpy
from natsort import natsorted #call natsorted
import os
import scanpy as sc
from scanpy.neighbors import neighbors
#from scanpy.neighbors import compute_mnn
#from scanpy.neighbors import compute_neighbors
from scanpy.external.pp import mnn_correct
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from numba import jit, cuda



try:
    from .SAE import SAE  # this is for installing package
except:
    from SAE import SAE  #  this is for testing whether DescModel work or not 
random.seed(201809)
np.random.seed(201809)
tf.random.set_seed(201809)
#tf.set_random_seed(201809) if tf.__version__<  "2.0" else tf.random.set_seed(201809)
#tf.set_random_seed(201809)
from sklearn.metrics import pairwise_distances
import math
from numpy import array

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')#the first parameter is shape and not name
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)

        call() 用来执行 Layer 的职能, 即当前 Layer 所有的计算过程均在该函数中完成
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q 

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ClusteringLayerGaussian(ClusteringLayer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super().__init__(n_clusters,weights,alpha,**kwargs)
    
    def call(self,inputs,**kwargs):
        sigma=1.0
        q=K.sum(K.exp(-K.square(K.expand_dims(inputs,axis=1)-self.clusters)/(2.0*sigma*sigma)),axis=2)
        q=K.transpose(K.transpose(q)/K.sum(q,axis=1))
        return q


class DescModel(object):
    """
    pretrain: 1.SAE
              2.kmeans
              3.model
              4.fit
              5.compile
    """

    def __init__(self,
                 dims,
                 obs,
                 x, # input matrix, row sample, col predictors 
                 alpha=1.0,
		 tol=0.005,
                 init='glorot_uniform', #initialization method
                 n_clusters= 10,  #kmeans clusters
                 louvain_resolution=1.0, # resolution for louvain 
                 n_neighbors=10,    # the 
                 pretrain_epochs=300, # epoch for autoencoder
                 epochs_fit=1, #epochs for each update,int or float 
                 batch_size=512, #batch_size for autoencoder
                 random_seed=201809,
		 activation='relu',
                 actincenter="tanh",# activation for the last layer in encoder, and first layer in the decoder 
                 drop_rate_SAE=0.2,
                 is_stacked=False,#True
                 use_earlyStop=True,
                 use_ae_weights=False,
		 save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp",
                 kernel_clustering="t"
                 # save result to save_dir, the default is "result_tmp". if recurvie path, the root dir must be exists, or there will be something wrong: for example : "/result_singlecell/dataset1" will return wrong if "result_singlecell" not exist
                 ):

        if not os.path.exists(save_dir):
            print("Create the directory:"+str(save_dir)+" to save result")
            os.mkdir(save_dir)
        self.dims = dims
        self.x=x #feature n*p, n:number of cells, p: number of genes
        self.obs=obs
        self.alpha = alpha
        self.tol=tol
        self.init=init
        self.n_clusters=n_clusters  #kmeans
        self.input_dim = dims[0]  # for clustering layer 
        self.n_stacks = len(self.dims) - 1
        self.is_stacked=is_stacked
        self.resolution=louvain_resolution
        self.n_neighbors=n_neighbors
        self.pretrain_epochs=pretrain_epochs
        self.epochs_fit=epochs_fit
        self.batch_size=batch_size
        self.random_seed=random_seed
        self.activation=activation
        self.actincenter=actincenter
        self.drop_rate_SAE=drop_rate_SAE
        self.is_stacked=is_stacked
        self.use_earlyStop=use_earlyStop
        self.use_ae_weights=use_ae_weights
        self.save_encoder_weights=save_encoder_weights
        self.save_encoder_step=save_encoder_step
        self.save_dir=save_dir
        self.kernel_clustering=kernel_clustering
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        #tf.set_random_seed(random_seed)
        #tf.set_random_seed(random_seed) if tf.__version__ < "2.0" else tf.random.set_seed(random_seed)
	    #pretrain autoencoder
        self.pretrain()
        


    def pretrain(self):
        sae=SAE(dims=self.dims,
        obs=self.obs,
        x=self.x,
		act=self.activation,
                drop_rate=self.drop_rate_SAE,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                actincenter=self.actincenter,
                init=self.init,
                use_earlyStop=self.use_earlyStop,
                save_dir=self.save_dir
           )
        # begin pretraining
        t0 = get_time()
        if self.use_ae_weights: 
            print("Checking whether %s  exists in the directory"%str(os.path.join(self.save_dir,'ae_weights.h5')))
            if not os.path.isfile(self.save_dir+"/ae_weights.h5"):
                print("The file ae_weights.h5 is not exits")
                if self.is_stacked:
                    sae.fit(self.x,epochs=self.pretrain_epochs)
                else:
                    sae.fit2(self.x,epochs=self.pretrain_epochs)
                self.autoencoder=sae.firstfit
                self.encoder=sae.finetune
                print("SAE finish")
            else:
                sae.autoencoders.load_weights(os.path.join(self.save_dir,"ae_weights.h5"))
                self.autoencoder=sae.firstfit
                self.encoder=sae.finetune
                print("SAE finish2")
        else:
            print("use_ae_weights=False, the program will rerun autoencoder")
            if self.is_stacked:
                sae.fit(self.x,epochs=self.pretrain_epochs)
            else:
                sae.fit2(self.x,epochs=self.pretrain_epochs)
            self.autoencoder=sae.firstfit
            self.encoder=sae.finetune
            #self.autoencoder=sae.autoencoders
            #self.encoder=sae.encoder
            print("SAE finish3")
        testdataout=sae.extract_feature(self.x)
        print("testdataout.shape",len(testdataout))
        print("testdataout[0].shape",np.asarray(testdataout[0]).shape)
        testdataout=np.asarray(testdataout[1])
        print("testdataout[1].shape",testdataout.shape)
        features11=pd.DataFrame(testdataout)
        features11.to_pickle(os.path.join(self.save_dir,"testdataout.npy"))
        adatas_test=sc.AnnData(testdataout,obs=self.obs[self.obs['data']=='testdata'])
        print("adatas_test",adatas_test)
        print("adatas_test.obs",adatas_test.obs_names)
        adatas_test.write_h5ad(os.path.join(self.save_dir,"adatas_test.h5ad"))


        tsnefeaturestestdataout=pd.DataFrame(adatas_test.X)#save autoencoder feature to do mytsne
        print('tsnefeaturestestdataout.shape',tsnefeaturestestdataout.shape)
        tsnefeaturestestdataout.to_pickle(os.path.join(self.save_dir,"tsnefeaturestestdataout.npy"))    
        
        print('Pretraining time is', get_time() - t0)
        #save ae results into disk
        if not os.path.isfile(os.path.join(self.save_dir,"ae_weights.h5")):
            self.autoencoder.save_weights(os.path.join(self.save_dir,'ae_weights.h5'))
            self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights.h5'))
            print('Pretrained weights are saved to %s /ae_weights.h5' % self.save_dir)
        #save autoencoder model
        self.autoencoder.save(os.path.join(self.save_dir,"autoencoder_model.h5"))
      
        ######################distance based graph###############
        t1 = get_time()
        
                               
        def kl_divergence(p, q):
            return entropy(p, q, base=None, axis=0) #try4
        
        def mymatric(XA,XB):
            #计算两组输入的每对之间的距离
            #print("CDIST XA",XA.ndim)
            #print("CDIST XA",XA)
            # jensenshannon
            return cdist([XA], [XB], metric=add)  #kl_divergence 越小表示两个分布越相似
        
        def add(p,q):
            return sum(p+q)
        
        #大小为(batch1的细胞数量 × batch2的细胞数量)，如果两个细胞是MNN，则对应的元素为True，否则为False。
        #['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p']
        
        kl_divergence = tf.keras.losses.KLDivergence()    
        def calculate_kl_2(row_A):
        # 计算当前行与矩阵B每一行的KL散度
            kl_row = tf.map_fn(lambda row_B: kl_divergence(row_A, row_B), batch2)
            return kl_row   
        def calculate_kl_1(row_A):
        # 计算当前行与矩阵batch1每一行的KL散度
            kl_row = tf.map_fn(lambda row_B: kl_divergence(row_A, row_B), batch1)
            return kl_row 

        def compute_mnn(batch1, batch2, k=5,p=1):
            t2=get_time() 
            if not os.path.isfile(self.save_dir+"/indices1.npy"):
                print("The file indices1.npy is not exits")
                print("batch1",batch1.shape)
                print("batch2",batch2.shape)
                
                # 计算batch1中每个细胞到batch2的最近邻
                nbrs1 = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',metric="euclidean").fit(batch2)   
            #metric=lambda batch1,batch2:mymatric(batch1,batch2)   
            #mymatric(batch1,batch2):ValueError: Metric not valid. Use sorted(sklearn.neighbors.VALID_METRICS['brute']) to get valid options. Metric can also be a callable function.
            #metric:If metric is a callable function, it takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors.
                distances1, indices1 = nbrs1.kneighbors(batch1)  #(n_queries, n_indexed)
                print("distances1",distances1)
                print("indices1",indices1.shape)
            
                pd.DataFrame(indices1).to_pickle(os.path.join(self.save_dir,"indices1.npy"))    
                pd.DataFrame(distances1).to_pickle(os.path.join(self.save_dir,"distances1.npy"))    


                # 计算batch2中每个细胞到batch1的最近邻
                nbrs2 = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',metric="euclidean").fit(batch1)
                distances2, indices2 = nbrs2.kneighbors(batch2)
                print("distances2",distances2.shape)
                print("indices2",indices2.shape)
                pd.DataFrame(indices2).to_pickle(os.path.join(self.save_dir,"indices2.npy"))    
                pd.DataFrame(distances2).to_pickle(os.path.join(self.save_dir,"distances2.npy"))    

                print("distance calculate finish")
                '''distances1 (10, 10)
            indices1 (10, 10)
            distances2 (20, 10)
            indices2 (20, 10)'''
            else:
                indices1=np.load(os.path.join(self.save_dir,"indices1.npy"),allow_pickle=True)
                indices2=np.load(os.path.join(self.save_dir,"indices2.npy"),allow_pickle=True)
                distances1=np.load(os.path.join(self.save_dir,"distances1.npy"),allow_pickle=True)
                distances2=np.load(os.path.join(self.save_dir,"distances2.npy"),allow_pickle=True)
            end3=get_time()
            print("distance_mnn time is",end3-t2)


            # 初始化MNN矩阵
            mnn = np.zeros((batch1.shape[0], batch2.shape[0]), dtype=bool)
            # 遍历batch1中的每个细胞
            for i in range(batch1.shape[0]):
                # 当前细胞在batch2中的最近邻索引(sc的最邻近st点的索引)
                knn1_indices = indices1[i]
                # 遍历batch2中的最近邻
                for neighbor in knn1_indices:
                    # 检查batch2中的最近邻是否在batch1的最近邻中
                    if i in indices2[neighbor]:
                        mnn[i, neighbor] = True
                    else:
                        #距离最近的batch2[j]，mnn[i,j]赋值true               
                        #print("distence[i]",distances1[i])  #distances1[i]按距离升序排列
                        #print("distances1[%s].min"%i,distances1[i].min())
                        if distances1[i].min()<p:
                            #j = distances1[i].argmin()   #输出distances1[i]的最小值索引都是0，因为它按照升序排列#sc_i点的最邻近st，距离最大的st索引
                            j=indices1[i][0]
                            #print("st_j",j)
                            mnn[i,j]=True
            return mnn
        # 输入矩阵
        sc_test=adatas_test[adatas_test.obs['batch']=='sc']
        print("sc_test1",sc_test.X.shape)
        print("sc_test1",type(sc_test.X))
        print("sc_test1",sc_test.X.ndim)
        st_test=adatas_test[adatas_test.obs['batch']=='st']
        print("st_test",st_test.X.shape)
        
        #####sc st 对应:MNN##############################################################
        #mnn
        #batch1=np.asarray(sc_test.X)[:10,:]
        #batch2=np.asarray(st_test.X)[:20,:]
        batch1=np.asarray(sc_test.X)
        batch2=np.asarray(st_test.X)
        #batch1=sc_test.X
        #batch2=st_test.X
        print("XA",type(batch1))
        print("XA.ndim",batch1.ndim)
        '''
        sc_sums = np.sum(np.asarray(sc_test.X), axis=1)
        sc_sums = sc_sums.reshape((40532,1))
        print("sc_sums",sc_sums.shape)
        st_sums = np.sum(np.asarray(st_test.X), axis=1)
        st_sums=st_sums.reshape((1,29870))
        print("st_sums.shape",st_sums.shape)
        add_matrix = np.dot(sc_sums, st_sums)
        '''
        #k,p
        mnn=compute_mnn(batch1, batch2,k=1000,p=0.5)
        print("mnn",mnn.shape)
        mnn=pd.DataFrame(mnn)
        mnn.index=sc_test.obs_names
        mnn.columns=st_test.obs_names
        mnn.to_pickle(os.path.join(self.save_dir,"mnn.npy"))    
        sc_test.write_h5ad(os.path.join(self.save_dir,"sc_test.h5ad"))
        st_test.write_h5ad(os.path.join(self.save_dir,"st_test.h5ad"))
        print('mnn time is', get_time() - t1)
        return sc_test


        '''
        # 计算每列的最大值
        column_max = tf.reduce_max(kl_similarity_matrix, axis=0)
        print("column_max",column_max)
        # 获取每列最大值对应的索引
        column_indices = tf.argmax(kl_similarity_matrix, axis=0)
        print("column_indices",column_indices)
        # 获取行列名
        row_names= pd.DataFrame(sc_test.obs_names)
        column_names= pd.DataFrame(st_test.obs_names)
        print("row_names",row_names)
        print("column_names",column_names)
        #row_names = tf.range(tf.shape(kl_similarity_matrix)[0])
        #column_names = tf.range(tf.shape(kl_similarity_matrix)[1])

        # 获取每列最大值对应的行名
        row_names_for_max = tf.gather(row_names, column_indices)   #从params的axis维根据indices的参数值获取切片 
        print("row_names_for_max",row_names_for_max)
        row_names_for_max=pd.DataFrame(row_names_for_max)
        row_names_for_max.to_pickle(os.path.join(self.save_dir,"row_names_for_max.npy"))
        # 打印每列的最大值及其对应的行列名
        for column, max_value, row_name in zip(column_names, column_max, row_names_for_max):
            print(f"Column {column}: Max Value = {max_value}, Row Name = {row_name}")
        
        sc_test.obs['st_index']=row_names_for_max
        sc_test.write_h5ad(os.path.join(self.save_dir,"sc_test.h5ad"))
        return sc_test
        '''

        ####################################################################
        
   
    #DEC的参数
    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        X_test=x[self.obs['data']=='testdata']
        print("X_test",X_test)
        return self.encoder.predict(X_test)  #self.encoder.predict(self.x=x=adata.X)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        '''q = self.model.predict(x, verbose=0)
        return q.argmax(1)
        
        """
        predict投入sc+st
        """
        q = self.model.predict(x, verbose=0)
        print('predict',q)  #q.argmax:AttributeError: 'list' object has no attribute 'argmax'
        print('pre_1',q[0])
        print('pre_1',q[0].shape)
        print('pre_1',q[0].argmax(1))
        return q[0].argmax(1),q[1].argmax(1)
        '''
        q = self.model.predict(x, verbose=0)
        return q[0].argmax(1),q[1].argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    

         

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='DescModel class test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--tol', default=0.005, type=float)
    parser.add_argument('--save_dir', default='result_tmp')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create mnist data to test 
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
    #from load_mnist import load_mnist
    x,y=load_mnist(sample_size=10000)
    init = 'glorot_uniform'
    #dims=[x.shape[-1], 500, 300, 100, 30]
    dims=[x.shape[-1],64,32]
    # prepare sample data to  the DESC model
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    desc = DescModel(dims=dims,x=x,louvain_resolution=0.8,use_ae_weights=True,epochs_fit=0.4)
    desc.model.summary()
    t0 = get_time()
    desc.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    Embedded_z,q_pred= desc.fit(maxiter=30)
    y_pred=q_pred.max(axis=1)
    obs_info=pd.DataFrame()
    obs_info["y_true"]=pd.Series(y.astype("U"),dtype="category")
    obs_info["y_pred"]=pd.Series(y_pred.astype("U"),dtype="category")
    adata=sc.AnnData(x,obs=obs_info)
    adata.obsm["X_Embeded_z"]=Embedded_z
    print('clustering time: ', (get_time() - t0))
    
    
