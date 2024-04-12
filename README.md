

# A Quick Tour of COSCST

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/shiy-shiy/SCST/">
    <img src="image/0201_FIG1.jpg" alt="Logo" width="800" height="600">
  </a>
  <h3 align="center">COSCST WORKFLOW</h3>
    <a href="https://github.com/shiy-shiy/SCST/">查看Demo</a>
    ·
    <a href="https://github.com/shiy-shiy/SCST/issues">报告Bug</a>
    ·
    <a href="https://github.com/shiy-shiy/SCST/issues">提出新特性</a>
  </p>

</p>

Single-cell RNA sequencing data excels in providing high sequencing depth and precision at the single-cell level, but lacks spatial information. Simultaneously, spatial transcriptomics technology visualizes gene expression patterns in their spatial context but has low resolution. Here, we present **COSCST** that combines these two datasets through autoencoder and supervised learning model to map single-cell RNA-seq data with spatial coordination and spatial transcriptomics with precise cell type annotation.

### 1. Installation

To install `COSCST` package you must make sure that your `tensorflow` version `2.x`. You decide to use CPU or GPU to run `tensorflow` according your devices. GPU could accelerate tensorflow by installing `tensorflow-gpu`. In addtation, please make sure your python version is compatible with tensorflow 2.x. In our paper, we used `python 3.6.x` .

We suggest using a separate conda environment for installing `cell2location`.

Create conda environment and install cell2location package

###### **Configuration requirements**

1. xxxxx x.x.x
2. xxxxx x.x.x

###### **INSTALL STEP**
```sh
git clone https://github.com/shiy-shiy / SCST.git
pip install COSCST
```

### 2. Usage
Firstly, import COSCST package.
```sh
from DIST import *
```

Secondly, create training and test dataset from outputs of COSCST network, following the /model/4_inputdata.ipynb, 4_testdata.ipynb
and 4_train_valid.ipynb.


Thirdly, run COSCST; transform the outputs of network into imputed expression matrix and its spot coordinate matrix.

```sh
data=sc.read_h5ad("testdata/inputdata_celltrek_kidney.h5ad")
data #202932 × 2977
#定义距离阈值，判断neighbor
save_dir="/loaddesc"
test1=desc.train(data,
        dims=[data.shape[1],512,64],
        myobs=data.obs,
        n_clusters=10,
        tol=0.1,#todo
        n_neighbors=10,
        batch_size=512,#
        epochs_fit=20,
        pretrain_epochs=50, #
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
```

where `dims`, `tol`, `batch_size`, `pretrain_epochs`,`learning_rate` are parameters of network, `use_GPU` is the GPUs whether you want to use.



### License

请阅读**CONTRIBUTING.md** 查阅为该项目做出贡献的开发者。

###Citation

### 作者

xxx@xxxx


 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/shiy-shiy / SCST/blob/master/LICENSE.txt)

### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [xxxxxxxxxxxxxx](https://connoratherton.com/loaders)

<!-- links -->
[your-project-path]:shiy-shiy / SCST
[contributors-shield]: https://img.shields.io/github/contributors/shiy-shiy / SCST.svg?style=flat-square
[contributors-url]: https://github.com/shiy-shiy / SCST/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shiy-shiy / SCST.svg?style=flat-square
[forks-url]: https://github.com/shiy-shiy / SCST/network/members
[stars-shield]: https://img.shields.io/github/stars/shiy-shiy / SCST.svg?style=flat-square
[stars-url]: https://github.com/shiy-shiy / SCST/stargazers
[issues-shield]: https://img.shields.io/github/issues/shiy-shiy / SCST.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shiy-shiy / SCST.svg
[license-shield]: https://img.shields.io/github/license/shiy-shiy / SCST.svg?style=flat-square
[license-url]: https://github.com/shiy-shiy / SCST/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian



