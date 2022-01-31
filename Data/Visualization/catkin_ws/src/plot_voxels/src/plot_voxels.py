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
# from geometry_msgs.msg import Point32
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
import cv2
import gc

x_dim = 128
y_dim = 128
z_dim = 8
grid_dims = np.array([x_dim, y_dim, z_dim])

# # R, G, B
# colors_2 = np.array([
#     (255, 255, 255), # None 0 -> 0
#     (70, 70, 70),    # Building 1 -> 1
#     (100, 40, 40),   # Fences 2 - > 2
#     (55, 90, 80),    # Other 3 -> 3
#     (255, 255, 0),   # Pedestrian 4->4
#     (153, 153, 153), # Pole 5 -> 5
#     (157, 234, 50),  # RoadLines 6 -> 5
#     (0, 0, 255),  # Road 7 -> 6
#     (255, 255, 255),  # Sidewalk 8 -> 6
#     (0, 155, 0),  # Vegetation 9 -> 9
#     (255, 0, 0),     # Vehicle 10 -> 10
#     (102, 102, 156), # Wall 11 -> 2
#     (220, 220, 0),   # TrafficSign 12 -> 5
#     (70, 130, 180),  # Sky 13 ->3
#     (255, 255, 255),     # Ground 14 -> 7
#     (150, 100, 100), # Bridge 15 -> 3
#     (230, 150, 140), # RailTrack 16 -> 3
#     (180, 165, 180), # GuardRail 17 -> 2
#     (250, 170, 30),  # TrafficLight 18 -> 5
#     (110, 190, 160), # Static 19 -> 3
#     (170, 120, 50),  # Dynamic 20 -> 3
#     (45, 60, 150),   # Water 21 -> 3
#     (145, 170, 100), # Terrain 22 -> 7
# ]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

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
        [-30, 30.0, 0.0],
        [-30, -30.0, 0.0],
        [30, 30.0, 0.0 ],
        [30.0, -30.0, 0.0]
    ]
    pub_MotionSC = rospy.Publisher('MotionSC_mapper', MarkerArray, queue_size=None)
    pub_LMSC = rospy.Publisher('LMSC_mapper', MarkerArray, queue_size=None)
    pub_SSC = rospy.Publisher('SSC_Full_mapper', MarkerArray, queue_size=None)
    pub_JS3C= rospy.Publisher('JS3C_mapper', MarkerArray, queue_size=None)
    # pub_MotionSC = rospy.Publisher('MotionSC_mapper', Marker, queue_size=None)
    # pub_LMSC = rospy.Publisher('LMSC_mapper', Marker, queue_size=None)
    # pub_SSC = rospy.Publisher('SSC_Full_mapper', Marker, queue_size=None)
    # pub_JS3C= rospy.Publisher('JS3C_mapper', Marker, queue_size=None)

    pub = [pub_MotionSC, pub_LMSC, pub_SSC, pub_JS3C]
    pub2 = rospy.Publisher('BEV', Image, queue_size=None)
    rospy.init_node('talker',disable_signals=True)
    #rate = rospy.Rate(0.1)
    #rospy.Duration(2)


    while not rospy.is_shutdown():

        MotionSC_markers = MarkerArray()
        LMSC_markers = MarkerArray()
        SSC_markers = MarkerArray()
        JS3C_markers = MarkerArray()
        # MotionSC_marker = Marker()
        # LMSC_marker = Marker()
        # SSC_marker = Marker()
        # JS3C_marker = Marker()
        # MotionSC_markers = []
        # LMSC_markers = []
        # SSC_markers = []
        # JS3C_markers = []
        markers = [MotionSC_markers, LMSC_markers, SSC_markers, JS3C_markers]

        data_dir = "../../Scenes/Cartesian/Test_Cartesian/Test/Town10_Heavy/cartesian"

        load_dir_MotionSC = os.path.join(data_dir, "MotionSC_11")
        load_dir_LMSC = os.path.join(data_dir, "LMSC_11")
        load_dir_SSC = os.path.join(data_dir, "SSC_Full_11")
        load_dir_JS3C = os.path.join(data_dir, "JS3CNet_11")
        load_dirs = [load_dir_MotionSC, load_dir_LMSC, load_dir_SSC, load_dir_JS3C]
        # load_dirs = [load_dir_MotionSC, load_dir_LMSC, load_dir_SSC]
        #print("load_dirs:",len(load_dirs))

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
        
        # valid_cells = counts >= 1 # 128x128x8
        # valid_x = xv[valid_cells]
        # valid_y = yv[valid_cells]
        # valid_z = zv[valid_cells]
        xv = xv.reshape(-1,) # 128x128x8 -> N
        yv = yv.reshape(-1,)
        zv = zv.reshape(-1,)
        
        model_labels = [None] * len(load_dirs) # index i has labels for model i

        start_time = time.time()
        for frame in range(int(num_files)):
            print("Elapsed time since last frame: ", time.time() - start_time)
            start_time = time.time()
            #MotionSC_markers.clear() 
            MotionSC_markers.markers.clear()
            LMSC_markers.markers.clear()
            SSC_markers.markers.clear()
            JS3C_markers.markers.clear()
            # MotionSC_markers = MarkerArray()
            # LMSC_markers = MarkerArray()
            # SSC_markers = MarkerArray()
            # JS3C_markers = MarkerArray()
            # bev
            img_filepath = os.path.join(load_bev, str(frame).zfill(6))
            print(img_filepath + ".jpg")

            img = cv2.imread(img_filepath + ".jpg")
            img = img[150:950, 525:1400]
            bridge = CvBridge()
            img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")
            
            # voxel
            # model_labels = [None] * len(load_dirs) # index i has labels for model i
            # model_markers = [] # imdex i has markers for model i

            # eval_filepath = os.path.join(load_eval, str(frame).zfill(6))
            #counts = np.fromfile(eval_filepath + ".bin", dtype="float32").reshape(grid_shape)

            #print("labels shape",labels.shape)
            
            #model_markers = [None] * len(load_dirs)

            # intervals = (max_dim - min_dim) / grid_dims
            # x = np.linspace(min_dim[0], max_dim[0], num=grid_dims[0]) + intervals[0] / 2
            # y = np.linspace(min_dim[1], max_dim[1], num=grid_dims[1]) + intervals[1] / 2
            # z = np.linspace(min_dim[2], max_dim[2], num=grid_dims[2]) + intervals[2] / 2
            # xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
            
            # # valid_cells = counts >= 1 # 128x128x8
            # # valid_x = xv[valid_cells]
            # # valid_y = yv[valid_cells]
            # # valid_z = zv[valid_cells]
            # xv = xv.reshape(-1,) # 128x128x8 -> N
            # yv = yv.reshape(-1,)
            # zv = zv.reshape(-1,)
            #labels = labels.reshape(-1,)
            # points = np.stack((xv, yv, zv), axis=1) # Nx3 
            
            for model in range (len(load_dirs)):
                points = np.stack((xv, yv, zv), axis=1) # Nx3 

                # intervals = (max_dim - min_dim) / grid_dims
                # x = np.linspace(min_dim[0], max_dim[0], num=grid_dims[0]) + intervals[0] / 2
                # y = np.linspace(min_dim[1], max_dim[1], num=grid_dims[1]) + intervals[1] / 2
                # z = np.linspace(min_dim[2], max_dim[2], num=grid_dims[2]) + intervals[2] / 2
                # xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
                
                # # valid_cells = counts >= 1 # 128x128x8
                # # valid_x = xv[valid_cells]
                # # valid_y = yv[valid_cells]
                # # valid_z = zv[valid_cells]
                # xv = xv.reshape(-1,) # 128x128x8 -> N
                # yv = yv.reshape(-1,)
                # zv = zv.reshape(-1,)
                # #labels = labels.reshape(-1,)
                # points = np.stack((xv, yv, zv), axis=1) # Nx3 
                #print('ps:',points.shape)

                frame_filepath = os.path.join(load_dirs[model], str(frame).zfill(6))
                #print(npoints.shape)
                model_labels[model] = np.fromfile(frame_filepath + ".label", dtype="uint32").reshape(grid_shape) # 128x128x8
                model_labels[model] = model_labels[model].reshape(-1,)
                #labels = np.fromfile(frame_filepath + ".label", dtype="uint32").reshape(grid_shape) # 128x128x8
                print("loading: ", frame_filepath + ".label")
                non_free = model_labels[model] != 0 # 128x128x8
                points = points[non_free, :]
                model_labels[model] = model_labels[model][non_free]

                # swap axes
                new_points = np.zeros(points.shape)
                new_points[:, 0] = points[:, 1]
                new_points[:, 1] = points[:, 0]
                new_points[:, 2] = points[:, 2]
                points = new_points
                theta = np.pi
                rotation_mtx = np.matrix([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0], [0,0,1]])
                #print(points[np.arange(0,points.shape[0], 3),:].shape,rotation_mtx.shape)
                #print(points[np.arange(0,points.shape[0], 3),:])
                #points[np.arange(0,points.shape[0], 3),:] = np.matmul(rotation_mtx,points[np.arange(0,points.shape[0], 3),:])
                #points = np.matmul(points,rotation_mtx)
                #print(points.shape)
                #print(model_labels[model].shape[0])

                ### Cube list
                marker = Marker()
                # if model == 0:
                #     marker = MotionSC_marker

                # elif model == 1:
                #     marker = LMSC_marker

                # elif model == 2:
                #     marker = SSC_marker

                # else:
                #     marker = JS3C_marker

                # marker_tmp = Marker()
                marker.id = model
                marker.ns = model_names[model] + "basic_shapes"
                marker.header.frame_id = "map"# change this to match model + scene name LMSC_000001
                marker.type = marker.CUBE_LIST
                # marker.action = marker.ADD
                if frame == 0:
                    marker.action = marker.ADD
                else:
                    marker.action = marker.MODIFY
                marker.lifetime.secs = 0
                marker.header.stamp = rospy.Time.now()

                # Pose
                # marker.pose.position.x = loc[0]
                # marker.pose.position.y = loc[1]
                # marker.pose.position.z = loc[2]
                # marker.pose.position.x = points[i,0] + model_offsets[model][0]
                # marker.pose.position.y = points[i,1] + model_offsets[model][1]
                # marker.pose.position.z = points[i,2] + model_offsets[model][2]
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1

                # Scale (meters)
                marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
                marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
                marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

                # if model == 3:
                #     model_labels[model][model_labels[model] == 7] = 6
                #     model_labels[model][model_labels[model] == 8] = 6
                #     model_labels[model][model_labels[model] == 11] = 2
                #     model_labels[model][model_labels[model] == 12] = 5
                #     model_labels[model][model_labels[model] == 13] = 3
                #     model_labels[model][model_labels[model] == 14] = 7
                #     model_labels[model][model_labels[model] == 15] = 3
                #     model_labels[model][model_labels[model] == 16] = 3
                #     model_labels[model][model_labels[model] == 17] = 2
                #     model_labels[model][model_labels[model] == 18] = 5
                #     model_labels[model][model_labels[model] == 19] = 3
                #     model_labels[model][model_labels[model] == 20] = 3
                #     model_labels[model][model_labels[model] == 21] = 3
                #     model_labels[model][model_labels[model] == 22] = 7
                #     model_labels[model][model_labels[model] > 22] = 0
                # pred = model_labels[model]
                # # point = Point()
                # #color = ColorRGBA()
                # point = point_s([points[:,0] + model_offsets[model][0], points[:,1] + model_offsets[model][1], points[:,2] + model_offsets[model][2]])
                # color = color_s([colors[pred][:,0], colors[pred][:,1], colors[pred][:,2], np.ones_like(colors[pred][:,0].shape)*0.75])
                # # point.x = points[:,0] + model_offsets[model][0]
                # # point.y = points[:,1] + model_offsets[model][1]
                # # point.z = points[:,2] + model_offsets[model][2]
                # #print(colors[pred][:,0].shape)
                # # color.r = colors[pred][:,0]
                # # color.g = colors[pred][:,1]
                # # color.b = colors[pred][:,2]
                # # color.r, color.g, color.b = colors[pred]
                # # color.a = np.ones_like(colors[pred][:,0].shape)*0.75
                # #marker.points.append(point)
                # #marker.colors.append(color)
                # marker.points = point
                # marker.colors = color
                # markers[model].markers.append(marker)
                points[:,0] += model_offsets[model][0]
                points[:,1] += model_offsets[model][1]
                points[:,2] += model_offsets[model][2]
                # if model == 0:
                #     MotionSC_marker.append(marker)
                # elif model == 1:
                #     LMSC_marker.append(marker)
                # elif model == 2:
                #     SSC_marker.append(marker)
                # else:
                #     JS3C_marker.append(marker)

                for i in range(model_labels[model].shape[0]):                   
                    pred = model_labels[model][i]

                    point = Point32()
                    #point = Point()
                    color = ColorRGBA()
                    # point.x = points[i,0] + model_offsets[model][0]
                    # point.y = points[i,1] + model_offsets[model][1]
                    # point.z = points[i,2] + model_offsets[model][2]
                    point.x = points[i,0]
                    point.y = points[i,1]
                    point.z = points[i,2]
                    color.r, color.g, color.b = colors[pred]

                    color.a = 0.75
                    marker.points.append(point)
                    marker.colors.append(color)
                #print(len(marker.points), len(marker.colors))
                #pub[model].publish(marker)
                # if model == 0:
                #     MotionSC_markers.markers.append(marker)
                # elif model == 1:
                #     SSC_markers.markers.append(marker)
                # elif model == 2:
                #     LMSC_markers.markers.append(marker)
                # else:
                #     JS3C_markers.markers.append(marker)
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


            # model_markers = [markers0, markers1, markers2]
            pub2.publish(img_msg)
            # pub_MotionSC.publish(MotionSC_marker)
            # pub_SSC.publish(SSC_marker)
            # pub_LMSC.publish(LMSC_marker)
            # pub_JS3C.publish(JS3C_marker)
            for model in range(len(load_dirs)):
                pub[model].publish(markers[model])
            #rospy.sleep(0.5)
            #gc.collect()
            # rospy.Rate(0.5)

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:        
        pass    

