import os              
os.environ['PYTHONHASHSEED'] = '0'
import desc          
import pandas as pd                                                    
import numpy as np                                                     
import scanpy as sc                                                                                 
from time import time                                                       
import sys
import matplotlib
import matplotlib.pyplot as plt
sc.settings.set_figure_params(dpi=300)
print(sys.version)
sc.logging.print_versions()
desc.__version__
import tensorflow as tf
tf.__version__
from scipy.io import mmwrite
import scipy
import anndata as ad
import random
import anndata

#self.encoder.load_weights('/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/2_desc/2_desc_222copy/sc_desc/encoder_weights_resolution_0.8_5.h5')

data=sc.read_h5ad("/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/4_predataset/inputdata_celltrek_kidney.h5ad")
data #202932 × 2977
#定义距离阈值，判断neighbor
save_dir="/home/huggs/shiyi/SCST/RCTD/RCTD/predata/celltrek_kidney/mymodel/5_newloaddesc/loaddesc"
test1=desc.train(data,
        dims=[data.shape[1],512,64],
        myobs=data.obs,
        n_clusters=10,
        tol=0.1,#todo
        n_neighbors=10,
        batch_size=512,#todo 512/1000
        epochs_fit=20,
        pretrain_epochs=50, #todo:1,300
        #louvain_resolution=[2.0],# not necessarily a list, you can only set one value, like, louvain_resolution=1.0
        save_dir=str(save_dir),
        do_mytsne=True,
        do_tsne=False,
        learning_rate=200, # the parameter of tsne
        use_GPU=True,#False
        num_Cores=4, #for reproducible, only use 1 cpu
        num_Cores_tsne=4,
        save_encoder_weights=True,
        save_encoder_step=3,# save_encoder_weights is False, this parameter is not used
        use_ae_weights=True,
        do_umap=False,
        do_myumap=True,
        do_myumap_X_Embeded_z=False) #if do_uamp is False, it will don't compute umap coordiate

test1.write_h5ad("loaddesc.h5ad")

#/home/huggs/anaconda3/envs/SCST/lib/python3.6/site-packages/desc/models