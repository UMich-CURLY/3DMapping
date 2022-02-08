# 3D Mapping
Welcome! This repository contains all software used to create CarlaSC (our dynamic scene completion data set) and MotionSC (our local semantic mapping network), and all baselines. Software documentation may be found in the wiki, and more information may be found on our [website](https://umich-curly.github.io/CarlaSC.github.io/). Below is an overview of our project. For more information on the data and method, see our website. For more information on the software and files involved, see the Wiki.

## Data: **CarlaSC**
Our data set is a novel data set with accurate, complete dynamic outdoor scenes for semantic scene completion or mapping. We use the CARLA simulator (1) to gather clean ground truth information, where we randomly place a large number of sensors in each scene, guaranteeing theoretical convergence to the true scene and network generalization to the full scene. Dynamic maps are difficult due to traces left behind by moving objects, and occlusions. Below is a comparison of a frame from our data set with a similar frame from Semantic KITTI (4). 

<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/21368455/153008032-d1332e4a-4872-4348-99ec-9fb64106e849.png">
</p>

## Networks: **MotionSC**
We create a network using ideas from MotionNet (2) and LMSCNet (3), which we call MotionSC. MotionSC uses temporal information to perform semantic scene completion in real-time, useful as a 3D local semantic map. We compare with baselines LMSCNet (3), JS3CNet (4), and SSCNet (6). A video comparison is shown below. Note that in the video, the ego vehicle is the stationary Tesla in the bottom right, and is excluded from the complete semantic scenes. 

<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/21368455/153005475-6ad63a00-b39e-477d-b887-07a3283fa14e.gif">
</p>

## Use 3DMapping
### Dependencies
* [Pytorch](https://pytorch.org/get-started/locally/) - we tested on PyTorch 1.10 and 1.8.2
* [SpConv](https://github.com/traveller59/spconv) - for running JS3CNet. We recommend using `spconv-cu113` and we found this is an issue with using `spconv-cu102` and cpu version.
* [Open3D](http://www.open3d.org/) - for visualizing map


## Acknowledgement
We utilize data and code from: 
- [1] CARLA Simulator https://carla.org/ 
- [2] MotionNet https://arxiv.org/abs/2003.06754 
- [3] LMSCNet https://arxiv.org/abs/2008.10559
- [4] Semantic KITTI https://arxiv.org/abs/1904.01416
- [5] JS3CNet https://arxiv.org/abs/2012.03762
- [6] SSCNet http://sscnet.cs.princeton.edu/

