# 3DMapping
Learning to map in 3D

## Log

***Nov 17, 2021*** 

* Added python class for cylindrical cells based on Cylinder3D implementation.
* Added Carla dataloader python script, currently generates 1000 red dummy data points using `load_dummy_point_cloud()` and plots blue points for cylindrical cell centroids that have data points associated to them.

## Use 3DMapping
### Dependencies
* [Pytorch](https://pytorch.org/get-started/locally/) - we tested on PyTorch 1.10 and 1.8.2
* [SpConv](https://github.com/traveller59/spconv) - we recommend using `spconv-cu113` and we found this is an issue with using `spconv-cu102` and cpu version.
* [Open3D](http://www.open3d.org/) - for visualizing map
### Get Started
### ToDo List
- [x] Run the model with dummy data
- [ ] Gather real data from simulator and start first training


## Acknowledgement
We utilize data and code from: 
- CARLA Simulator https://carla.org/ 
- MotionNet https://arxiv.org/abs/2003.06754 