# 3D Mapping
Welcome! This repository contains all software used to create CarlaSC (our dynamic scene completion data set) and MotionSC (our local semantic mapping network), and all baselines. Below is an overview of our project. For more information on the data and method, see our [website](https://umich-curly.github.io/CarlaSC.github.io/). For more information on the software and files involved, see the [Wiki](https://github.com/UMich-CURLY/3DMapping/wiki).

## Table of Contents
 - [Data](#data-carlasc)
 - [Networks](#networks-motionsc)
 - [Use 3DMapping](#use-3dmapping)
   - [Dependencies](#dependencies)
 - [Acknowledgement](#acknowledgement)

## Data: **CarlaSC**
A novel data set with accurate, complete dynamic outdoor scenes for semantic scene completion or mapping. We use the [CARLA simulator](https://carla.org/) (1) to gather clean ground truth information, where we randomly place a large number of sensors in each scene, guaranteeing theoretical convergence to the true scene and network generalization to the full scene. Dynamic maps are difficult due to traces left behind by moving objects, and occlusions. Below is a comparison of a frame from our data set with a similar frame from [Semantic KITTI](http://www.semantic-kitti.org/index.html) (4). 

More introduction about CarlaSC dataset and download links are avialable on our [website](https://umich-curly.github.io/CarlaSC.github.io/).
<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/21368455/153008032-d1332e4a-4872-4348-99ec-9fb64106e849.png">
</p>

## Networks: **MotionSC**
We create a network using ideas from MotionNet (2) and LMSCNet (3), which we call MotionSC. MotionSC uses temporal information to perform semantic scene completion in real-time, useful as a 3D local semantic map. We compare with baselines LMSCNet (3), JS3CNet (5), and SSCNet (6). A video comparison is shown below. Note that in the video, the ego vehicle is the stationary Tesla in the bottom right, and is excluded from the complete semantic scenes. 

<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/21368455/153005475-6ad63a00-b39e-477d-b887-07a3283fa14e.gif">
</p>

## Use 3DMapping
### Dependencies
* [Pytorch](https://pytorch.org/get-started/locally/) - we tested on PyTorch 1.10 and 1.8.2
* [Open3D](http://www.open3d.org/) - for visualizing map
* [Spconv](https://github.com/traveller59/spconv) - Only required by JS3CNet. The JS3CNet is a little complicated on its dependency, please check our forked [JS3CNet](https://github.com/Song-Jingyu/JS3C-Net) to run it.
* For data generation and visualization tasks, you can find the dependencies at [Wiki Home](https://github.com/UMich-CURLY/3DMapping/wiki) and [Data Visualization](https://github.com/UMich-CURLY/3DMapping/wiki/Data-Visualization).

### Data Generation and Visualization
We provide script to generate the dataset. We also provide scripts for visualizing the data. We have detailed explanation on the useful scripts and parameters in our [wiki](https://github.com/UMich-CURLY/3DMapping/wiki). You can check the `TODO` block in scripts for parameters that could be adjusted.

### Training and Testing
We provide synthetic scripts of training and testing MotionSC, LMSCNet and SSCNet in our repo. You can check the `TODO` block in scripts for parameters that could be adjusted. You can find how to use them on our [wiki](https://github.com/UMich-CURLY/3DMapping/wiki). Due to special dependencies of JS3CNet, you can check our forked [JS3CNet](https://github.com/Song-Jingyu/JS3C-Net) to run it. 

## Acknowledgement
We utilize data and code from: 
- [1] CARLA Simulator https://carla.org/ 
- [2] MotionNet https://arxiv.org/abs/2003.06754 
- [3] LMSCNet https://arxiv.org/abs/2008.10559
- [4] Semantic KITTI https://arxiv.org/abs/1904.01416
- [5] JS3CNet https://arxiv.org/abs/2012.03762
- [6] SSCNet http://sscnet.cs.princeton.edu/

## Reference
If you find our work useful in your research work, consider citing [our paper](paper_link)!
<!-- TODO! Paper bibtext -->