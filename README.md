# 3D Mapping
Welcome! This repository contains all software used to create CarlaSC (our dynamic scene completion data set) and MotionSC (our local semantic mapping network), and all baselines. Software documentation may be found in the wiki, and more information may be found on our website. Below is an overview of our project.

## Data: **CarlaSC**
<p align="center">
  <img width="740" src="https://user-images.githubusercontent.com/21368455/153003213-03659261-45c5-46f5-8177-b5482e7c0604.png">
</p>

## Networks: **MotionSC**
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
- CARLA Simulator https://carla.org/ 
- MotionNet https://arxiv.org/abs/2003.06754 
- LMSCNet https://arxiv.org/abs/2008.10559
- JS3CNet https://arxiv.org/abs/2012.03762
