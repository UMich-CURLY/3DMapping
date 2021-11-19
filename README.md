# 3DMapping
Learning to map in 3D

We utilize data and code from: 
- CARLA Simulator https://carla.org/ 
- MotionNet https://arxiv.org/abs/2003.06754 

## Log

***Nov 17, 2021*** 

* Added python class for cylindrical cells based on Cylinder3D implementation.
* Added Carla dataloader python script, currently generates 1000 red dummy data points using `load_dummy_point_cloud()` and plots blue points for cylindrical cell centroids that have data points associated to them.
