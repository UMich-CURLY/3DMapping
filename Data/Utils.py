import numpy as np

# Test labels
IDX_CAT_MAP = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'other'}
CAT_IDX_MAP = {'bg': 0, 'bus': 1, 'ped': 2, 'other': 3}

# Actual labels
LABEL_COLORS = np.array([
    (255, 255, 255), # None (free) 0
    (70, 70, 70),    # Building 1
    (100, 40, 40),   # Fences 2
    (55, 90, 80),    # Other 3
    (220, 20, 60),   # Pedestrian 4
    (153, 153, 153), # Pole 5
    (157, 234, 50),  # RoadLines 6
    (128, 64, 128),  # Road 7
    (244, 35, 232),  # Sidewalk 8
    (107, 142, 35),  # Vegetation 9
    (0, 0, 142),     # Vehicle 10
    (102, 102, 156), # Wall 11
    (220, 220, 0),   # TrafficSign 12
    (70, 130, 180),  # Sky 13
    (81, 0, 81),     # Ground 14
    (150, 100, 100), # Bridge 15
    (230, 150, 140), # RailTrack 16
    (180, 165, 180), # GuardRail 17
    (250, 170, 30),  # TrafficLight 18
    (110, 190, 160), # Static 19
    (170, 120, 50),  # Dynamic 20
    (45, 60, 150),   # Water 21
    (145, 170, 100), # Terrain 22
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses