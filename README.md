# 3D Mapping
Welcome! This repository contains all software used to create CarlaSC (our dynamic scene completion data set) and MotionSC (our local semantic mapping network), and all baselines. Below is an overview of our project. For more information on the data and method, see our [website](https://umich-curly.github.io/CarlaSC.github.io/). For more information on the software and files involved, see the [Wiki](https://github.com/UMich-CURLY/3DMapping/wiki).

## Table of Contents
 - [Data](#data-carlasc)
 - [Networks](#networks-motionsc)
 - [Use 3DMapping](#use-3dmapping)
   - [Dependencies](#dependencies)
 - [Results](#results)
   - [CarlaSC](#carlasc)
   - [SemanticKITTI](#semantickitti)
 - [Acknowledgement](#acknowledgement)

## Data: **CarlaSC**
A novel data set with accurate, complete dynamic outdoor scenes for semantic scene completion or mapping. We use the [CARLA simulator](https://carla.org/) (1) to gather clean ground truth information, where we randomly place a large number of sensors in each scene, guaranteeing theoretical convergence to the true scene and network generalization to the full scene. Dynamic maps are difficult due to traces left behind by moving objects, and occlusions. Below is a comparison of a frame from our data set with a similar frame from [Semantic KITTI](http://www.semantic-kitti.org/index.html) (4). 

More introduction about CarlaSC dataset and download links are avialable on our [website](https://umich-curly.github.io/CarlaSC.github.io/).
<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/91337470/175836944-f4c91add-95bc-490e-aaee-248005bf1351.png">
</p>

## Networks: **MotionSC**
We create a network using ideas from MotionNet (2) and LMSCNet (3), which we call MotionSC. MotionSC extends semantic scene completion to achieve a higher level of scene understanding by integrating the structure of a view-volume scene completion network with the temporal backbone of an object tracking network. It is built off the idea that semantic scene completion is fundamentally a similar task to 3D semantic mapping, where the major difference is the use of temporal information, readily available in robotic applications. We compare with scene completion baselines LMSCNet (3), JS3CNet (5), and SSCNet (6). We show that the number of past scans (T) is correlated with improvements in semantic and geometric completeness quantitatively and qualitatively. Note that in the video below, the ego vehicle is the stationary Tesla in the bottom right, and is excluded from the complete semantic scenes. 

<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/91337470/175836958-8e08f5d7-2017-476e-a826-803e983cffa9.gif">
</p>

## Use 3DMapping
### Dependencies
* [Pytorch](https://pytorch.org/get-started/locally/) - we tested on PyTorch 1.10 and 1.8.2
* [Open3D](http://www.open3d.org/) - for visualizing map
* [Spconv](https://github.com/traveller59/spconv) - Only required by JS3CNet. The JS3CNet is a little complicated on its dependency, please check our forked [JS3CNet](https://github.com/Song-Jingyu/JS3C-Net) to run it.
* For data generation and visualization tasks, you can find the dependencies at [Wiki Home](https://github.com/UMich-CURLY/3DMapping/wiki) and [Data Visualization](https://github.com/UMich-CURLY/3DMapping/wiki/Data-Visualization).

We also provide a environment.yml which you can use to create a conda environment
```
conda env create -f environment.yaml
conda activate neuralblox
```

### Data Generation and Visualization
We provide script to generate the dataset. We also provide scripts for visualizing the data. We have detailed explanation on the useful scripts and parameters in our [wiki](https://github.com/UMich-CURLY/3DMapping/wiki). You can check the `TODO` block in scripts for parameters that could be adjusted.

### Training and Testing
We provide synthetic scripts of training and testing MotionSC, LMSCNet and SSCNet in our repo. You can check the `TODO` block in scripts for parameters that could be adjusted. You can find how to use them on our [wiki](https://github.com/UMich-CURLY/3DMapping/wiki). Due to special dependencies of JS3CNet, you can check our forked [JS3CNet](https://github.com/Song-Jingyu/JS3C-Net) to run it. 

## Results
We trained our model on the CarlaSC dataset and the SemanticKitti dataset. Note that while improved semantic scene completion networks are being released with better results, our method of incorporating temporal information may be applied to these networks for enhanced mapping capabilities. 

### CarlaSC
For the MotionSC model on CarlaSC we also list its performance dependent on the amount of temporal information provided, T.

#### Inference Time (Measured on NVIDIA GeForce RTX 3090)

| Method          | Latency (ms) |
|-----------------|:------------:|
| LMSCNet SS      |     4.86     |
| SSCNet Full     |     2.18     |
| JS3C-Net        |     166.2    |
| **MotionSC (T=1)**  |     5.72     |

#### Semantic Completeness

| Method          | Mean IoU | Accuracy |  Free | Building | Barrier | Other | Pedestrian |  Pole |  Road | Ground | Sidewalk | Vegetation | Vehicles |
|-----------------|:--------:|:--------:|:-----:|:--------:|:-------:|:-----:|:----------:|:-----:|:-----:|:------:|:--------:|:----------:|:--------:|
| LMSCNet SS      |   42.53  |   94.64  | 97.41 |   25.61  |   3.35  | 11.31 |    33.76   | 43.54 | 85.96 |  21.15 |   52.64  |    39.99   |   53.09  |
| SSCNet Full     |   41.91  |   94.11  | 96.02 |   27.04  |   1.82  | 13.65 |    29.69   | 27.02 | 88.45 |  25.89 |   65.36  |    33.29   |   52.78  |
| JS3C-Net        |   48.95  |   95.48  | 96.78 |   34.68  |   3.03  | 22.94 |    43.64   | 44.50 | 93.31 |  30.90 |   75.15  |    34.35   |   59.21  |
| **MotionSC (T=1)**  |   46.31  |   95.11  | 97.42 |   31.59  |   2.63  | 14.77 |    39.87   | 42.11 | 90.57 |  25.89 |   60.77  |    42.41   |   61.37  |
| **MotionSC (T=5)**  |   45.35  |   95.00  | 97.43 |   29.48  |   2.54  | 17.48 |    41.87   | 43.43 | 90.90 |  22.08 |   58.43  |    35.79   |   59.41  |
| **MotionSC (T=10)** |   47.01  |   95.15  | 97.44 |   32.29  |   2.35  | 19.82 |    44.06   | 45.47 | 90.19 |  27.35 |   62.48  |    36.92   |   58.80  |
| **MotionSC (T=16)** |   47.45  |   95.57  | 97.60 |   34.91  |   2.66  | 22.86 |    37.78   | 43.87 | 90.12 |  28.31 |   66.20  |    41.59   |   56.08  |

#### Geometric Completeness

| Method          | Precision | Recall |  IoU |
|-----------------|:--------:|:--------:|:-----:|
| LMSCNet SS      |   95.62  |   98.95  | 85.98 |
| SSCNet Full     |   85.87  |   93.05  | 80.69 |
| JS3C-Net        |   89.43  |   93.02  | 83.80 |
| **MotionSC (T=1)**  |   93.32  |   92.16  | 86.46 |
| **MotionSC (T=5)**  |   94.76  |   90.57  | 86.25 |
| **MotionSC (T=10)** |   93.17  |   92.43  | 86.56 |
| **MotionSC (T=16)** |   94.61  |   91.77  | 87.21 |

### SemanticKITTI
For the SemanticKITTI dataset, we only compare with T=1 as the rules of the competion require a single input frame. Our results may be found on the Semantic KITTI leader board under user "tigeriv4." The results of other models are collected from their papers.

#### Semantic Completeness

| Method         | Mean IoU | Accuracy |  Road | Sidewalk | Parking | Other-ground | Building |  Car  | Truck | Bicycle | Motorcycle | Other-vehicle | vegetation | Trunk | Terrain | Person | Bicyclist | Motorcyclist | Fence |  Pole | Traffic-sign |
|----------------|:--------:|:--------:|:-----:|:--------:|:-------:|:------------:|:--------:|:-----:|:-----:|:-------:|:----------:|:-------------:|:----------:|:-----:|:-------:|:------:|:---------:|:------------:|:-----:|:-----:|:------------:|
| LMSCNet SS     |   17.62  |   56.72  | 64.80 |   34.68  |  29.02  |     4.62     |   38.08  | 30.89 |  1.47 |    0    |      0     |      0.81     |    41.31   | 19.89 |  32.05  |    0   |     0     |       0      | 21.32 | 15.01 |     0.84     |
| SSCNet Full    |   16.14  |   49.98  | 51.15 |   30.76  |  27.12  |     6.44     |   34.53  | 24.26 |  1.18 |   0.54  |    0.78    |      4.34     |    35.25   | 18.17 |  29.01  |  0.25  |    0.25   |     0.03     | 19.87 | 13.10 |     6.73     |
| JS3C-Net       |   23.8   |   56.6   |  64.7 |   39.9   |   34.9  |     14.1     |   37.4   |  33.3 |  7.2  |   14.4  |     8.8    |      12.7     |    43.1    |  19.6 |   40.5  |   8.0  |    5.1    |      0.4     |  30.4 |  18.9 |     15.9     |
| **MotionSC (T=1)** |   18.4   |   56.9   |  66.0 |   36.5   |   29.6  |      7.0     |   39.0   |  31.4 |  1.0  |    0    |      0     |      3.6      |    40.0    |  19.0 |   30.0  |    0   |     0     |       0      |  23.4 |  20.0 |      3.4     |

## Acknowledgement
We utilize data and code from: 
- [1] CARLA Simulator https://carla.org/ 
- [2] SemanticKITTI http://www.semantic-kitti.org/
- [3] MotionNet https://arxiv.org/abs/2003.06754 
- [4] LMSCNet https://arxiv.org/abs/2008.10559
- [5] Semantic KITTI https://arxiv.org/abs/1904.01416
- [6] JS3CNet https://arxiv.org/abs/2012.03762
- [7] SSCNet http://sscnet.cs.princeton.edu/

## Reference
If you find our work useful in your research work, consider citing [our paper](https://arxiv.org/abs/2203.07060)!
```
@misc{https://doi.org/10.48550/arxiv.2203.07060,
  doi = {10.48550/ARXIV.2203.07060},
  
  url = {https://arxiv.org/abs/2203.07060},
  
  author = {Wilson, Joey and Song, Jingyu and Fu, Yuewei and Zhang, Arthur and Capodieci, Andrew and Jayakumar, Paramsothy and Barton, Kira and Ghaffari, Maani},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {MotionSC: Data Set and Network for Real-Time Semantic Mapping in Dynamic Environments},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
