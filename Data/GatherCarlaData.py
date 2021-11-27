#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Open3D Lidar visuialization example for CARLA"""

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
import copy
from matplotlib import cm
import open3d as o3d

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


# Records velocities of non-ego vehicles, pose of lidar, lidar scan with semantic labels
def semantic_lidar_callback(point_cloud, world, lidar_id, vehicle_id, save_dir, view=0):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    non_ego = np.array(data['ObjIdx']) != vehicle_id

    points = np.array([data['x'], data['y'], data['z']]).T
    points = points[non_ego, :]

    # Unique tags
    actor_velocities = []
    actor_list = world.get_actors()
    lidar_loc = actor_list.find(lidar_id).get_transform()
    tags = np.unique(np.array(data['ObjIdx']))
    for id in tags:
        if id == 0:
            continue
        actor = actor_list.find(int(id))
        actor_vel = actor.get_velocity()
        # Transform to lidar
        actor_vel = np.asarray([actor_vel.x, actor_vel.y, actor_vel.z])
        to_world = np.array(actor.get_transform().get_inverse_matrix())[:3, :3]
        to_lidar = np.array(lidar_loc.get_matrix())[:3, :3]
        rel_vel = np.matmul(to_lidar, np.matmul(to_world, actor_vel))

        actor_velocities.append([id, rel_vel[0], rel_vel[1], rel_vel[2]])

    # Record time, velocities, position
    labels = np.array(data['ObjTag'])[non_ego]
    if view == 0:
        print(np.unique(labels, return_counts=True))
    instances = np.array(data['ObjIdx'])[non_ego]
    actor_velocities = np.array(actor_velocities)

    location = np.array(lidar_loc.get_matrix())

    timestamp = world.get_snapshot().timestamp
    frame = world.get_snapshot().frame

    # Record time, velocities, pose, points, labels, instances
    time_file = open(save_dir + '/times' + str(view) + '.txt', 'a')
    time_file.write(str(frame) + ", " + str(timestamp) + "\n")
    time_file.close()

    np.save(save_dir + "/velocities" + str(view) + "/" + str(frame), actor_velocities)
    np.save(save_dir + "/instances" + str(view) + "/" + str(frame), instances)
    np.save(save_dir + "/labels" + str(view) + "/" + str(frame), labels)
    np.save(save_dir + "/velodyne" + str(view) + "/" + str(frame), points)
    np.save(save_dir + "/pose" + str(view) + "/" + str(frame), location)


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    # Semantic lidar
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')

    # Not semantic
    #     lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    #     if arg.no_noise:
    #         lidar_bp.set_attribute('dropoff_general_rate', '0.0')
    #         lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
    #         lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
    #     else:
    #         lidar_bp.set_attribute('noise_stddev', '0.2')

    lidar_bp.set_attribute('upper_fov', str(arg["upper_fov"]))
    lidar_bp.set_attribute('lower_fov', str(arg["lower_fov"]))
    lidar_bp.set_attribute('channels', str(arg["channels"]))
    lidar_bp.set_attribute('range', str(arg["range"]))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg["points_per_second"]))
    return lidar_bp


def main(arg):
    """Main function of the script"""
    client = carla.Client(arg["host"], arg["port"])
    client.set_timeout(2.0)
    world = client.get_world()

    # Record
    client.start_recorder(arg["storage_dir"] + "/recording01.log")

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # How often to update
        delta = 0.1

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg["no_rendering"]
        world.apply_settings(settings)

        # Create a car
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(arg["filter"])[0]
        vehicle_transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(True)

        # Create semantic lidar
        NUM_SENSORS = 20
        views = np.arange(NUM_SENSORS)
        lidars = []
        for i in range(NUM_SENSORS):
            os.mkdir(arg["storage_dir"] + "/velocities" + str(i))
            os.mkdir(arg["storage_dir"] + "/instances" + str(i))
            os.mkdir(arg["storage_dir"] + "/labels" + str(i))
            os.mkdir(arg["storage_dir"] + "/velodyne" + str(i))
            os.mkdir(arg["storage_dir"] + "/pose" + str(i))

            if i == 0: # Onboard sensor
                offsets = [-0.5, 0.0, 1.8]
            else:
                offsets = np.random.uniform([-20, -20, 1], [20, 20, 10], [3,])

            lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)
            # Location of lidar, fixed to vehicle
            user_offset = vehicle_transform.location
            lidar_transform = carla.Transform(carla.Location(x=offsets[0], y=offsets[1], z=offsets[2]) + user_offset)
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            lidars.append(lidar)
            print(lidar_transform)
            # Add callback
            lidar.listen(lambda data, view=views[i]:
                         semantic_lidar_callback(data, world, lidar.id, vehicle.id, arg["storage_dir"], view=view))

        # # Create semantic lidar number two (different view)
        # lidar_bp2 = generate_lidar_bp(arg, world, blueprint_library, delta)
        # # Location of lidar, fixed to vehicle
        # user_offset2 = carla.Location(0.0, 0.0, 0.0)
        # lidar_transform2 = carla.Transform(carla.Location(x=1.5, y=5.0, z=3.8) + user_offset2)
        # lidar2 = world.spawn_actor(lidar_bp2, lidar_transform2, attach_to=vehicle)
        # # Add callback
        # lidar2.listen(lambda data: semantic_lidar_callback(data, world, lidar2.id, vehicle.id, arg["storage_dir"],
        #                                                    view=1))

        frame = 0
        dt0 = datetime.now()
        while True:
            time.sleep(0.005)
            world.tick()

            process_time = datetime.now() - dt0
            # sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()) + "\n")
            # sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

    finally:
        client.stop_recorder()

        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        vehicle.destroy()

        for lidar in lidars:
            lidar.destroy()


if __name__ == "__main__":
    args = {
        "host": "localhost",
        "port": 2000,
        "no_rendering": False,
        "noise": False,
        "filter": "model3",
        "upper_fov": 15,
        "lower_fov": -25,
        "channels": 64.0,
        "range": 50,
        "points_per_second": 500000,
        "show_axis": True,
        "storage_dir": "/home/tigeriv/Data/Carla/02"
    }

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')