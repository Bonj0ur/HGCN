# Learning spatial interaction representation with Heterogeneous Graph Convolution Networks for urban land-use inference
Pytorch implementation of our paper “Learning spatial interaction representation with Heterogeneous Graph Convolution Networks for urban land-use inference”. We propose a novel framework, Heterogeneous Graph Convolution Networks (HGCN) to explicitly account for the spatial demand and supply components embedded in spatial interaction data, in addition to local characteristic features, for urban land-use inference.

![Pipeline](assets\Framework.png)

## 1. Requirements
- pytorch
- numpy
- pandas
- sklearn

## 2. Dataset
We provide [London Dataset](./data/london_data/) in this repository.

| File Name      | Content                                       | Dim       |
| :------------- | :-------------------------------------------- | :-------- |
| df_zonevec.csv | Doc2Vec representations for spatial units     | [786,72]  |
| df_LUarea.csv  | Urban land-use distribution for spatial units | [786,8]   |
| bike_od_mx.npy | Origin-destination flow matrix                | [786,786] |
| ls_edge.pkl    | Adjacency edge list                           | [786,786] |
| df_dis.csv     | Physical distance between spatial units       | [786,786] |

Due to commercial/legal restriction, we  provide [mock data](./data/shenzhen_data/) for Shenzhen Dataset in this repository. Due to capacity limitation, we provide the data download address [[HGCN (figshare.com)](https://figshare.com/s/001e24dda6ff9f840b56)].

| File Name          | Content                                            | Dim           |
| :----------------- | :------------------------------------------------- | :------------ |
| df_zonevec.csv     | Doc2Vec representations for spatial units          | [12345,72]    |
| df_area.csv        | Mock urban land-use distribution for spatial units | [12345,8]     |
| flow_mx.npy        | Mock origin-destination flow matrix                | [12345,12345] |
| adj.npy            | Adjacency matrix                                   | [12345,12345] |
| wdis_normalize.csv | Normalized distance-decay between spatial units    | [12345,12345] |

## 3. Reproducibility of the results

You can reproduce the results in London according to the following instructions. 

### 3.1 Comparison of models

#### 3.1.1 Results in the paper

![Comparison of models](assets\Comparison_of_models.png)

#### 3.1.2 Code

```bash
############################### Group A ################################
# Multi-Layer Perception
python train.py --k-fold 30 --train_size 0.7 --train_mode 1 --dataset 1
############################### Group B ################################
# GCN (Distance-decay)
python train.py --k-fold 30 --train_size 0.7 --train_mode 2 --dataset 1
# GCN (In-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 3 --dataset 1
# GCN (Out-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 4 --dataset 1
# GCN (Average-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 5 --dataset 1
# GCN (Adjacency)
python train.py --k-fold 30 --train_size 0.7 --train_mode 6 --dataset 1
############################### Group C ################################
# Bi-GCN (In-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 7 --dataset 1
# Bi-GCN (Average-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 8 --dataset 1
# Bi-GCN (Out-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 9 --dataset 1
############################### Group D ################################
# Tri-GCN
python train.py --k-fold 30 --train_size 0.7 --train_mode 10 --dataset 1
############################### Group E ################################
# Adj-Bi-GCN (In-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 11 --dataset 1
# Adj-Bi-GCN (Average-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 12 --dataset 1
# Adj-Bi-GCN (Out-mobility)
python train.py --k-fold 30 --train_size 0.7 --train_mode 13 --dataset 1
# Adj-Tri-GCN
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1
```

### 3.2 Vector dimensions for spatial interaction and dependence features

#### 3.2.1 Results in the paper

![Vector dimensions for spatial interaction and dependence features](assets\Vector_dimensions_for_spatial_interaction_and_dependence_features.png)

#### 3.2.2 Code

```bash
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 62
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 60
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 58
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 56
                                   ......
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 8
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 6
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 4
python train.py --k-fold 30 --train_size 0.7 --train_mode 14 --dataset 1 --proportion 2
```

### 3.3 Size of training set

#### 3.3.1 Results in the paper（KL Divergence）

![Size of training set](assets\Size_of_training_set.png)

#### 3.3.2 Code

You only need to change the train_size (e.g. 0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9) in the following code to reproduce all the results in the paper. 

```bash
############################### Group A ################################
# Multi-Layer Perception
python train.py --k-fold 30 --train_size 0.1 --train_mode 1 --dataset 1
############################### Group B ################################
# GCN (Distance-decay)
python train.py --k-fold 30 --train_size 0.1 --train_mode 2 --dataset 1
# GCN (In-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 3 --dataset 1
# GCN (Out-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 4 --dataset 1
# GCN (Average-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 5 --dataset 1
# GCN (Adjacency)
python train.py --k-fold 30 --train_size 0.1 --train_mode 6 --dataset 1
############################### Group C ################################
# Bi-GCN (In-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 7 --dataset 1
# Bi-GCN (Average-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 8 --dataset 1
# Bi-GCN (Out-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 9 --dataset 1
############################### Group D ################################
# Tri-GCN
python train.py --k-fold 30 --train_size 0.1 --train_mode 10 --dataset 1
############################### Group E ################################
# Adj-Bi-GCN (In-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 11 --dataset 1
# Adj-Bi-GCN (Average-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 12 --dataset 1
# Adj-Bi-GCN (Out-mobility)
python train.py --k-fold 30 --train_size 0.1 --train_mode 13 --dataset 1
# Adj-Tri-GCN
python train.py --k-fold 30 --train_size 0.1 --train_mode 14 --dataset 1
```

### 3.4 Results in Shenzhen

You only need to change the dataset ID [--dataset 2] (1: for London dataset; 2: for Shenzhen dataset), then you can run the code for Shenzhen dataset.

```bash
python train.py --k-fold 30 --train_size XX --train_mode XX --dataset 2
```

