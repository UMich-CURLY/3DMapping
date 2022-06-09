import os
import numpy as np
import pdb
import torch
#from visualization_msgs.msg import *
#from geometry_msgs.msg import Point32
#from std_msgs.msg import ColorRGBA

CLASS_COUNTS_REMAPPED = np.array([
    0,         # Marked as zero because void is filtered out 
    0,   
    257352935,    
    64627941,          
    16752,          
    224016,
    0,          
    75812,          
    0,          
    0,          
    0,      
    539944,
    10538966,          
    1543817,   
    158171883,     
    9730727,     
    3474123,     
    1478073,
    5743794,     
    3345519,  
    0
], dtype=np.long)


LABELS_REMAP = np.array([
    0,  #void
    1,  #dirt
    0,  #none
    2,  #grass
    3,  #tree
    4,  #pole
    5,  #water
    6,  #sky
    7,  #vehicle
    8,  #object
    9,  #asphalt
    0,  # None
    10, #building
    0,  
    0,
    11, #log
    0,
    12, #person
    13, #fence
    14, #bush
    0,
    0,
    0,
    15, #concrete
    0,
    0,
    0,
    16, #barrier
    0,
    0,
    0,
    17, #puddle
    0,
    18, #mud
    19, #34rubble
    0,
    0,
    0,
    0,
    0,
    20  #free space (Added in later)
], dtype=np.uint8)

DYNAMIC_LABELS = np.array([
    0, 20 #7, 8, 12
])

colors = np.array([ # RGB
    (0,0,0),        #0void
    (108, 64, 20),  #1dirt
    (0,102,0),      #2grass
    (0,255,0),      #3tree
    (0,153,153),    #4pole
    (0,128,255),    #5water
    (0,0,255),      #6sky    
    (255,255,0),    #7vehicle
    (255,0,127),    #8object
    (64,64,64),     #9asphalt
    (255,0,0),      #10building
    (102,0,0),      #11log
    (204,153,255),  #12person
    (102, 0, 204),  #13fence
    (255,153,204),  #14bush
    (170,170,170),  #15concrete
    (41,121,255),   #16barrier
    (134,255,239),  #17puddle
    (99,66,34),     #18mud
    (110,22,138),   #19rubble
    (255,255,255)   #20freespace
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses

"""
def publish_voxels(map, pub, centroids, min_dim, 
    max_dim, grid_dims, model="DiscreteBKI", pub_dynamic=False,
    valid_voxels_mask=None):
    '''
    Publishes voxel map over ros to be visualized in rviz
    Input:
        map: HxWxDxC voxel map where H=height, W=width,
            D=depth, C=num_classes 
        pub: rospy publisher handle
        centroids: H*W*Dx2 indices from centroids of voxel map
        min_dim: 3x1 minimum dimensions in xyz
        max_dim 3x1 maximum dimensions in xyz
        grid_dims: 3x1 voxel grid resolution in xyz
        model: name of model used (Default: DiscreteBKI)
    '''
    if map.dim()==4:
        semantic_map = torch.argmax(map / torch.sum(map, dim=-1, keepdim=True), dim=-1, keepdim=True)
        semantic_map = semantic_map.reshape(-1, 1)
    else:
        semantic_map = map.reshape(-1, 1)

    if valid_voxels_mask!=None:
        valid_voxels_mask = valid_voxels_mask.reshape(-1)

    # Remove dynamic labels if specified
    if not pub_dynamic:
        dynamic_class = torch.tensor([
            0,
            7,
            8,
            12,
            20
        ], device=semantic_map.device).reshape(1, -1)
        # pdb.set_trace()
        dynamic_mask = torch.all(
            semantic_map.ne(dynamic_class), dim=-1
        )
        # pdb.set_trace()
        centroids = centroids[dynamic_mask]
        semantic_map = semantic_map[dynamic_mask]
  
        if valid_voxels_mask!=None:
            valid_voxels_mask = valid_voxels_mask[dynamic_mask]

    # Only publish nonfree voxels
    if valid_voxels_mask!=None:
        centroids = centroids[valid_voxels_mask]
        semantic_map = semantic_map[valid_voxels_mask].reshape(-1, 1)

    next_map = MarkerArray()

    marker = Marker()
    marker.id = 0
    marker.ns = model + "semantic_map"
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    
    # if frame == 0:
    #     marker.action = marker.ADD
    # else:
    #     marker.action = marker.MODIFY
    marker.lifetime.secs = 0
    # marker.header.stamp = 0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]
    
    for i in range(semantic_map.shape[0]):              
        pred = semantic_map[i]

        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]

        color.r, color.g, color.b = colors[pred]

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)
    
    next_map.markers.append(marker)

    pub.publish(next_map)

def publish_pc(pc, labels, pub, min_dim, 
    max_dim, grid_dims, model="DiscreteBKI", pub_dynamic=False):
    '''
    Publishes voxel map over ros to be visualized in rviz

    Input:
        map: HxWxDxC voxel map where H=height, W=width,
            D=depth, C=num_classes 
        pub: rospy publisher handle
        points: H*W*Dx2 indices from points of map
        min_dim: 3x1 minimum dimensions in xyz
        max_dim 3x1 maximum dimensions in xyz
        grid_dims: 3x1 voxel grid resolution in xyz
        model: name of model used (Default: DiscreteBKI)
    '''

    # Only publish nonfree voxels
    nonfree_mask = (labels!=LABELS_REMAP[0]) & (labels!=LABELS_REMAP[-1])

    nonfree_points = pc[nonfree_mask]
    nonfree_labels = labels[nonfree_mask].reshape(-1, 1)

    # Remove dynamic labels if specified
    if not pub_dynamic:
        dynamic_labels = torch.from_numpy(DYNAMIC_LABELS).to(pc.device)
        dynamic_mask = torch.all(torch.ne(nonfree_labels, dynamic_labels), dim=-1)
        nonfree_points = nonfree_points[dynamic_mask]
        nonfree_labels = nonfree_labels[dynamic_mask].reshape(-1)
        

    next_map = MarkerArray()

    marker = Marker()
    marker.id = 0
    marker.ns = model + "semantic_map"
    marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    # marker.header.stamp = 0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]
    
    for i in range(nonfree_labels.shape[0]):              
        pred = nonfree_labels[i].cpu().numpy().astype(np.uint32)

        point = Point32()
        color = ColorRGBA()
        point.x = nonfree_points[i, 0]
        point.y = nonfree_points[i, 1]
        point.z = nonfree_points[i, 2]

        # pdb.set_trace()
        color.r, color.g, color.b = colors[pred]

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)
    
    next_map.markers.append(marker)
    pub.publish(next_map)
"""


