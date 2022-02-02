#!/usr/bin/env python

from matplotlib import markers
import rospy
import numpy as np
import time
import os
import json
import pdb
from visualization_msgs.msg import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import gc

x_dim = 128
y_dim = 128
z_dim = 8
grid_dims = np.array([x_dim, y_dim, z_dim])

# 11 Classes
colors = np.array([
    (255, 255, 255), # 0 None
    (70, 70, 70),    # 1 Building
    (100, 40, 40),   # 2 Fences
    (55, 90, 80),    # 3 Other
    (255, 255, 0),   # 4 Pedestrian
    (153, 153, 153), # 5 Pole
    (0, 0, 255),     # 6 Road
    (255, 255, 255), # 7 Ground
    (255, 255, 255), # 8 Sidewalk
    (0, 155, 0),     # 9 Vegetation
    (255, 0, 0),     # 10 Vehicle

]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

def point_s(self):
    point_s.x = self[0]
    point_s.y = self[1]
    point_s.z = self[2]
    return point_s

def color_s(self):
    color_s.r = self[0]
    color_s.g = self[1]
    color_s.b = self[2]
    color_s.a = self[2]
    return color_s

def talker():
    model_names = ['MotionSC', 'LMSC', 'SSC_Full', 'JS3C']
    model_offsets = [
        # [-35, 35.0, 0.0],
        # [-35, -35.0, 0.0],
        # [35, 35.0, 0.0 ],
        # [35.0, -35.0, 0.0]
        [-30, 29.0, 0.0],
        [-30, -28.5, 0.0],
        [30, 29.0, 0.0 ],
        [30.0, -28.5, 0.0]
    ]
    pub_MotionSC = rospy.Publisher('MotionSC_mapper', MarkerArray, queue_size=10)
    pub_LMSC = rospy.Publisher('LMSC_mapper', MarkerArray, queue_size=10)
    pub_SSC = rospy.Publisher('SSC_Full_mapper', MarkerArray, queue_size=10)
    pub_JS3C= rospy.Publisher('JS3C_mapper', MarkerArray, queue_size=10)

    pub = [pub_MotionSC, pub_LMSC, pub_SSC, pub_JS3C]
    pub2 = rospy.Publisher('BEV', Image, queue_size=10)
    rospy.init_node('talker',disable_signals=True)


    # while not rospy.is_shutdown():

    MotionSC_markers = MarkerArray()
    LMSC_markers = MarkerArray()
    SSC_markers = MarkerArray()
    JS3C_markers = MarkerArray()

    markers = [MotionSC_markers, LMSC_markers, SSC_markers, JS3C_markers]

    data_dir = "../../Scenes/Cartesian/Test_Cartesian/Test/Town10_Heavy/cartesian"

    load_dir_MotionSC = os.path.join(data_dir, "MotionSC_11")
    load_dir_LMSC = os.path.join(data_dir, "LMSC_11")
    load_dir_SSC = os.path.join(data_dir, "SSC_Full_11")
    load_dir_JS3C = os.path.join(data_dir, "JS3CNet_11")
    load_dirs = [load_dir_MotionSC, load_dir_LMSC, load_dir_SSC, load_dir_JS3C]
    print("load_dirs:", load_dirs)

    load_bev = os.path.join(data_dir, "bev")

    load_eval = os.path.join(data_dir, "evaluation")
    
    with open(os.path.join(load_eval, "params.json") ) as f:
        params = json.load(f)
        grid_shape = [params["num_channels"]] + list(params["grid_size"])
        grid_shape = [int(i) for i in grid_shape][1:]
        min_dim = np.array(params["min_bound"])
        max_dim = np.array(params["max_bound"])

    sequence = sorted(os.listdir(load_dirs[0]))
    num_files = (len(sequence)-1) / 2

    intervals = (max_dim - min_dim) / grid_dims
    x = np.linspace(min_dim[0], max_dim[0], num=grid_dims[0]) + intervals[0] / 2
    y = np.linspace(min_dim[1], max_dim[1], num=grid_dims[1]) + intervals[1] / 2
    z = np.linspace(min_dim[2], max_dim[2], num=grid_dims[2]) + intervals[2] / 2
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    
    xv = xv.reshape(-1,) # 128x128x8 -> N
    yv = yv.reshape(-1,)
    zv = zv.reshape(-1,)
    
    model_labels = [None] * len(load_dirs) # index i has labels for model i

    start_time = time.time()
    total_time = 0
    for frame in range(int(num_files)):
        # print("Elapsed time since last frame: ", time.time() - start_time)
        total_time += time.time() - start_time
        start_time = time.time()
        #MotionSC_markers.clear() 
        MotionSC_markers.markers.clear()
        LMSC_markers.markers.clear()
        SSC_markers.markers.clear()
        JS3C_markers.markers.clear()

        # bev
        img_filepath = os.path.join(load_bev, str(frame).zfill(6))

        img = cv2.imread(img_filepath + ".jpg")
        img = img[150:950, 525:1400]
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
        
        for model in range (len(load_dirs)):
            points = np.stack((xv, yv, zv), axis=1) # Nx3 

            frame_filepath = os.path.join(load_dirs[model], str(frame).zfill(6))
            model_labels[model] = np.fromfile(frame_filepath + ".label", dtype="uint32").reshape(grid_shape) # 128x128x8
            model_labels[model] = model_labels[model].reshape(-1,)

            non_free = model_labels[model] != 0 # 128x128x8
            points = points[non_free, :]
            model_labels[model] = model_labels[model][non_free]

            # swap axes
            new_points = np.zeros(points.shape)
            new_points[:, 0] = points[:, 1]
            new_points[:, 1] = points[:, 0]
            new_points[:, 2] = points[:, 2]
            points = new_points

            ### Cube list
            marker = Marker()
            marker.id = model
            marker.ns = model_names[model] + "basic_shapes"
            marker.header.frame_id = "map"# change this to match model + scene name LMSC_000001
            marker.type = marker.CUBE_LIST
            marker.action = marker.ADD
            if frame == 0:
                marker.action = marker.ADD
            else:
                marker.action = marker.MODIFY
            marker.lifetime.secs = 0
            marker.header.stamp = rospy.Time.now()

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1

            # Scale (meters)
            marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
            marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
            marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

            points[:,0] += model_offsets[model][0]
            points[:,1] += model_offsets[model][1]
            points[:,2] += model_offsets[model][2]

            for i in range(model_labels[model].shape[0]):                   
                pred = model_labels[model][i]

                point = Point32()
                color = ColorRGBA()
                point.x = points[i,0]
                point.y = points[i,1]
                point.z = points[i,2]
                color.r, color.g, color.b = colors[pred]

                color.a = 0.75
                marker.points.append(point)
                marker.colors.append(color)

            markers[model].markers.append(marker)

            ### individual Cube 
            # for i in range(model_labels[model].shape[0]): # Move initialization of points out of loop
                
            #     marker = Marker()
            #     marker.id = i
            #     # points[i] = np.matmul(rotation_mtx,points[i])

            #     #loc = points[i,:]
            #     #print(loc.shape)
            #     pred = model_labels[model][i]
            #     marker.ns = model_names[model] + "basic_shapes"
            #     marker.header.frame_id = "map"# change this to match model + scene name LMSC_000001
            #     marker.type = marker.CUBE
            #     if i == 0:
            #         marker.action = marker.ADD
            #     else:
            #         marker.action = marker.MODIFY
            #     marker.lifetime.secs = 5
            #     marker.header.stamp = rospy.Time.now()

            #     # Pose
            #     # marker.pose.position.x = loc[0]
            #     # marker.pose.position.y = loc[1]
            #     # marker.pose.position.z = loc[2]
            #     marker.pose.position.x = points[i,0] + model_offsets[model][0]
            #     marker.pose.position.y = points[i,1] + model_offsets[model][1]
            #     marker.pose.position.z = points[i,2] + model_offsets[model][2]
            #     marker.pose.orientation.x = 0.0
            #     marker.pose.orientation.y = 0.0
            #     marker.pose.orientation.z = 0.0
            #     marker.pose.orientation.w = 1

            #     # Color
            #     marker.color.r, marker.color.g, marker.color.b = colors[pred]
            #     marker.color.a = 0.75

            #     # Scale (meters)
            #     marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
            #     marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
            #     marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

            #     markers[model].markers.append(marker)

        pub2.publish(img_msg)
        for model in range(len(load_dirs)):
            pub[model].publish(markers[model])
        gc.collect()

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:      
        pass    

