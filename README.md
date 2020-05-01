Hyperbolic Graph Convolutional Networks in PyTorch
==================================================

## 1. Overview

This repository is a graph representation learning library, containing an implementation of Hyperbolic Graph Convolutions [[1]](http://web.stanford.edu/~chami/files/hgcn.pdf) in PyTorch, as well as multiple embedding approaches including:

#### Shallow methods (```Shallow```)

  * Shallow Euclidean
  * Shallow Hyperbolic [[2]](https://arxiv.org/pdf/1705.08039.pdf)
  * Shallow Euclidean + Features (see [[1]](http://web.stanford.edu/~chami/files/hgcn.pdf))
  * Shallow Hyperbolic + Features (see [[1]](http://web.stanford.edu/~chami/files/hgcn.pdf))
  
#### Neural Network (NN) methods 

  * Multi-Layer Perceptron (```MLP```)
  * Hyperbolic Neural Networks (```HNN```) [[3]](https://arxiv.org/pdf/1805.09112.pdf)
  
#### Graph Neural Network (GNN)  methods 

  * Graph Convolutional Neural Networks (```GCN```) [[4]](https://arxiv.org/pdf/1609.02907.pdf)
  * Graph Attention Networks (```GAT```) [[5]](https://arxiv.org/pdf/1710.10903.pdf)
  * Hyperbolic Graph Convolutions (```HGCN```) [[1]](http://web.stanford.edu/~chami/files/hgcn.pdf)

All models can be trained for 

  * Link prediction (```lp```)
  * Node classification (```nc```)

## 2. Setup

### 2.1 Installation with conda

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```git clone https://github.com/HazyResearch/hgcn```

```cd hgcn```

```conda env create -f environment.yml```

### 2.2 Installation with pip

Alternatively, if you prefer to install dependencies with pip, please follow the instructions below:

```virtualenv -p [PATH to python3.7 binary] hgcn```

```source hgcn/bin/activate```

```pip install -r requirements.txt```

### 2.3 Datasets

The ```data/``` folder contains source files for:

  * Cora
  * Pubmed
  * Disease 
  * Airport

To run this code on new datasets, please add corresponding data processing and loading in ```load_data_nc``` and ```load_data_lp``` functions in ```utils/data_utils.py```.

## 3. Usage

### 3.1 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 

### 3.2  ```train.py```

This script trains models for link prediction and node classification tasks. 
Metrics are printed at the end of training or can be saved in a directory by adding the command line argument ```--save=1```.

```
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --dropout DROPOUT     dropout probability
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --epochs EPOCHS       maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        l2 regularization strength
  --optimizer OPTIMIZER
                        which optimizer to use, can be any of [Adam,
                        RiemannianAdam]
  --momentum MOMENTUM   momentum in optimizer
  --patience PATIENCE   patience for early stopping
  --seed SEED           seed for training
  --log-freq LOG_FREQ   how often to compute print train/val metrics (in
                        epochs)
  --eval-freq EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --save SAVE           1 to save model and logs and 0 otherwise
  --save-dir SAVE_DIR   path to save training logs and model weights (defaults
                        to logs/task/date/run/)
  --sweep-c SWEEP_C
  --lr-reduce-freq LR_REDUCE_FREQ
                        reduce lr every lr-reduce-freq or None to keep lr
                        constant
  --gamma GAMMA         gamma for lr scheduler
  --print-epoch PRINT_EPOCH
  --grad-clip GRAD_CLIP
                        max norm for gradient clipping, or None for no
                        gradient clipping
  --min-epochs MIN_EPOCHS
                        do not early stop before min-epochs
  --task TASK           which tasks to train on, can be any of [lp, nc]
  --model MODEL         which encoder to use, can be any of [Shallow, MLP,
                        HNN, GCN, GAT, HGCN]
  --dim DIM             embedding dimension
  --manifold MANIFOLD   which manifold to use, can be any of [Euclidean,
                        Hyperboloid, PoincareBall]
  --c C                 hyperbolic radius, set to None for trainable curvature
  --r R                 fermi-dirac decoder parameter for lp
  --t T                 fermi-dirac decoder parameter for lp
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        path to pretrained embeddings (.npy file) for Shallow
                        node classification
  --pos-weight POS_WEIGHT
                        whether to upweight positive class in node
                        classification tasks
  --num-layers NUM_LAYERS
                        number of hidden layers in encoder
  --bias BIAS           whether to use bias (1) or not (0)
  --act ACT             which activation function to use (or None for no
                        activation)
  --n-heads N_HEADS     number of attention heads for graph attention
                        networks, must be a divisor dim
  --alpha ALPHA         alpha for leakyrelu in graph attention networks
  --use-att USE_ATT     whether to use hyperbolic attention in HGCN model
  --double-precision DOUBLE_PRECISION
                        whether to use double precision
  --dataset DATASET     which dataset to use
  --val-prop VAL_PROP   proportion of validation edges for link prediction
  --test-prop TEST_PROP
                        proportion of test edges for link prediction
  --use-feats USE_FEATS
                        whether to use node features or not
  --normalize-feats NORMALIZE_FEATS
                        whether to normalize input node features
  --normalize-adj NORMALIZE_ADJ
                        whether to row-normalize the adjacency matrix
  --split-seed SPLIT_SEED
                        seed for data splits (train/test/val)
```

## 4. Examples

We provide examples of training commands used to train HGCN and other graph embedding models for link prediction and node classification. In the examples below, we used a fixed random seed set to 1234 for reproducibility purposes. Note that results might slightly vary based on the machine used. To reproduce results in the paper, run each commad for 10 random seeds and average the results.

### 4.1 Training HGCN

#### Link prediction

  * Cora (Test ROC-AUC=93.79): 

```python train.py --task lp --dataset cora --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

  * Pubmed (Test ROC-AUC: 95.17):

```python train.py --task lp --dataset pubmed --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0``` 

  * Disease (Test ROC-AUC: 87.14):

```python train.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PoincareBall --normalize-feats 0 --log-freq 5```

  * Airport (Test ROC-AUC=97.43):

```python train.py --task lp --dataset airport --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.0 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0 --c None```

#### Node classification

  * Cora and Pubmed: 

To train train a HGCN node classification model on Cora and Pubmed datasets, pre-train embeddings for link prediction as decribed in the previous section. Then train a MLP classifier using the pre-trained embeddings (```embeddings.npy``` file saved in the ```save-dir``` directory). For instance for the Pubmed dataset:
 
```python train.py --task nc --dataset pubmed --model Shallow --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0.0005 --manifold Euclidean --log-freq 5 --cuda 0 --use-feats 0 --pretrained-embeddings [PATH_TO_EMBEDDINGS]```

  * Disease (Test accuracy: 76.77):

```python train.py --task nc --dataset disease_nc --model HGCN --dim 16 --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0```

### 4.2 Train other graph embedding models

#### Link prediction on the Cora dataset

 * Shallow Euclidean (Test ROC-AUC=86.40): 

```python train.py --task lp --dataset cora --model Shallow --manifold Euclidean --lr 0.01 --weight-decay 0.0005 --dim 16 --num-layers 0 --use-feats 0 --dropout 0.2 --act None --bias 0 --optimizer Adam --cuda 0```

 * Shallow Hyperbolic (Test ROC-AUC=85.97): 

```python train.py --task lp --dataset cora --model Shallow --manifold PoincareBall --lr 0.01 --weight-decay 0.0005 --dim 16 --num-layers 0 --use-feats 0 --dropout 0.2 --act None --bias 0 --optimizer RiemannianAdam --cuda 0```

 * GCN (Test ROC-AUC=89.22): 

```python train.py --task lp --dataset cora --model GCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.2 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 0```

 * HNN (Test ROC-AUC=90.79): 

```python train.py --task lp --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.2 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1```

#### Node classification on the Pubmed dataset

 * HNN (Test accuracy=68.20): 
 
``` python train.py --task nc --dataset pubmed --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0```

 * MLP (Test accuracy=73.00):
  
```python train.py --task nc --dataset pubmed --model MLP --lr 0.01 --dim 16 --num-layers 2 --act None --bias 0 --dropout 0.2 --weight-decay 0.001 --manifold Euclidean --log-freq 5 --cuda 0```

 * GCN (Test accuracy=78.30): 
 
```python train.py --task nc --dataset pubmed --model GCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.7 --weight-decay 0.0005 --manifold Euclidean --log-freq 5 --cuda 0```

 * GAT (Test accuracy=78.50): 

```python train.py --task nc --dataset pubmed --model GAT --lr 0.01 --dim 16 --num-layers 2 --act elu --bias 1 --dropout 0.5 --weight-decay 0.0005 --alpha 0.2 --n-heads 4 --manifold Euclidean --log-freq 5 --cuda 0```

## Citation

If you find this code useful, please cite the following paper: 
```
@inproceedings{chami2019hyperbolic,
  title={Hyperbolic graph convolutional neural networks},
  author={Chami, Ines and Ying, Zhitao and R{\'e}, Christopher and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4869--4880},
  year={2019}
}
```

## Some of the code was forked from the following repositories

 * [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn)
 * [gae](https://github.com/tkipf/gae/tree/master/gae)
 * [hyperbolic-image-embeddings](https://github.com/KhrulkovV/hyperbolic-image-embeddings)
 * [pyGAT](https://github.com/Diego999/pyGAT)
 * [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)
 * [geoopt](https://github.com/geoopt/geoopt)

## References

[1] [Chami, I., Ying, R., Ré, C. and Leskovec, J. Hyperbolic Graph Convolutional Neural Networks. NIPS 2019.](http://web.stanford.edu/~chami/files/hgcn.pdf)

[2] [Nickel, M. and Kiela, D. Poincaré embeddings for learning hierarchical representations. NIPS 2017.](https://arxiv.org/pdf/1705.08039.pdf)

[3] [Ganea, O., Bécigneul, G. and Hofmann, T. Hyperbolic neural networks. NIPS 2017.](https://arxiv.org/pdf/1805.09112.pdf)

[4] [Kipf, T.N. and Welling, M. Semi-supervised classification with graph convolutional networks. ICLR 2017.](https://arxiv.org/pdf/1609.02907.pdf)

[5] [Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. Graph attention networks. ICLR 2018.](https://arxiv.org/pdf/1710.10903.pdf)
