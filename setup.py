from distutils.core import setup
setup(
  name = 'COSCST',         # How you named your package folder (foo)
  packages = ['COSCST'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Single-cell RNA sequencing data excels in providing high sequencing depth and precision at the single-cell level, but lacks spatial information. Simultaneously, spatial transcriptomics technology visualizes gene expression patterns in their spatial context but has low resolution. Here, we present COSCST that combines these two datasets through autoencoder and supervised learning model to map single-cell RNA-seq data with spatial coordination and spatial transcriptomics with precise cell type annotation. ',   # Give a short description about your library
  author = 'Yi Shi ,Gang Hu',                   # Type in your name
  author_email = 'shiyi@nankai.mail.edu.cn, huggs@nankai.edu.cn',      # Type in your E-Mail
  url = 'https://github.com/shiy-shiy/SCST/',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/Yiguan/Cookie2Dict/archive/master.zip',
  keywords = ['single cell', 'spatial transcriptome'],   # Keywords that define your package best
  install_requires=['matplotlib>=2.2'
				#'pydot', 
				'tensorflow',
				#'keras==2.1', 
				'scanpy',
				'louvain',
				'python-igraph',  
				'h5py',
				'pandas', ],
  classifiers=[
    'Development Status :: 3 - Alpha',
  	'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
