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
from queue import Queue
from queue import Empty
from carla import VehicleLightState as vls
import logging

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# Records velocities of non-ego vehicles, pose of lidar, lidar scan with semantic labels
def semantic_lidar_callback(point_cloud, world, lidar_id, vehicle_id, save_dir, frame, view=0):
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
    instances = np.array(data['ObjIdx'])[non_ego]
    actor_velocities = np.array(actor_velocities)

    location = np.array(lidar_loc.get_matrix())

    timestamp = world.get_snapshot().timestamp

    # Record time, velocities, pose, points, labels, instances
    time_file = open(save_dir + '/times' + str(view) + '.txt', 'a')
    time_file.write(str(frame) + ", " + str(timestamp) + "\n")
    time_file.close()

    np.save(save_dir + "/velocities" + str(view) + "/" + str(frame), actor_velocities)
    np.save(save_dir + "/instances" + str(view) + "/" + str(frame), instances)
    np.save(save_dir + "/labels" + str(view) + "/" + str(frame), labels)
    np.save(save_dir + "/velodyne" + str(view) + "/" + str(frame), points)
    np.save(save_dir + "/pose" + str(view) + "/" + str(frame), location)


def bev_camera_callback(image, world, save_dir, frame):
    image.save_to_disk(save_dir + "/bev/" + str(frame) + ".jpg")


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


def create_traffic(world, client, num_vehicle, traffic_manager):
    # Create actors
    blueprints = get_actor_blueprints(world, "vehicle.*", "All")
    blueprintsWalkers = get_actor_blueprints(world, "walker.pedestrian.*", '2')
    # Make sure safe
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
    blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    # Spawn
    batch = []
    vehicles_list = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    # Imports
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    # Create vehicles
    for n, transform in enumerate(spawn_points):
        if n >= num_vehicle:  # Number vehicles
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        if True:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                     .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, True):  # not 100% sure
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    return vehicles_list


def main(arg):
    """Main function of the script"""
    client = carla.Client(arg["host"], arg["port"])
    client.set_timeout(2.0)
    world = client.get_world()

    if not os.path.exists(arg["storage_dir"]):
        os.mkdir(arg["storage_dir"])

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

        spectator = world.get_spectator()
        # To ensure synchrony
        lidar_queue = Queue()
        camera_queue = Queue()

        vehicles_list = create_traffic(world, client, arg["num_vehicles"], traffic_manager)

        # Create semantic lidar
        NUM_SENSORS = 20
        views = np.arange(NUM_SENSORS)
        lidars = []
        for i in range(NUM_SENSORS):
            if not os.path.exists(arg["storage_dir"] + "/velocities" + str(i)):
                os.mkdir(arg["storage_dir"] + "/velocities" + str(i))
            if not os.path.exists(arg["storage_dir"] + "/instances" + str(i)):
                os.mkdir(arg["storage_dir"] + "/instances" + str(i))
            if not os.path.exists(arg["storage_dir"] + "/labels" + str(i)):
                os.mkdir(arg["storage_dir"] + "/labels" + str(i))
            if not os.path.exists(arg["storage_dir"] + "/velodyne" + str(i)):
                os.mkdir(arg["storage_dir"] + "/velodyne" + str(i))
            if not os.path.exists(arg["storage_dir"] + "/pose" + str(i)):
                os.mkdir(arg["storage_dir"] + "/pose" + str(i))

            if i == 0: # Onboard sensor
                offsets = [-0.5, 0.0, 1.8]
            else:
                offsets = np.random.uniform([-10, -10, 1], [10, 10, 5], [3,])
                print(offsets)

            lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)
            # Location of lidar, fixed to vehicle
            lidar_transform = carla.Transform(carla.Location(x=offsets[0], y=offsets[1], z=offsets[2]))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            lidars.append(lidar)
            # Add callback

            # FOR LOOP Breaks
            lidars[i].listen(lambda data, view=views[i]: lidar_queue.put([data, view]))

        # Camera
        if not os.path.exists(arg["storage_dir"] + "/bev"):
            os.mkdir(arg["storage_dir"] + "/bev")

        cameras = []
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        cam_location = carla.Location(0, 0, 50)
        cam_rotation = carla.Rotation(270, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        cam01 = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        cam01.listen(camera_queue.put)
        cameras.append(cam01)

        frame = 0
        while True:
            world.tick()
            spectator.set_transform(vehicle.get_transform())
            time.sleep(0.005)
            # Store Data
            try:
                for _ in range(len(lidars)):
                    data, view = lidar_queue.get()
                    print(frame, view)
                    semantic_lidar_callback(data, world, lidars[view].id, vehicle.id, arg["storage_dir"], frame, view=view)

            except Empty:
                print("    Dropped LIDAR data")
            try:
                for _ in range(len(cameras)):
                    image = camera_queue.get()
                    bev_camera_callback(image, world, arg["storage_dir"], frame)

            except Empty:
                print("    Dropped Camera data")

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
        "storage_dir": "/home/tigeriv/Data/Carla/Data/Scenes/04/raw",
        "num_vehicles": 50
    }

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
