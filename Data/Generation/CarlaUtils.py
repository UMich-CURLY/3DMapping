import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
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

    # Add noise (0.2 centimeters)
    points += np.random.uniform(-0.02, 0.02, size=points.shape)

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
        actor_vel = np.asarray([actor_vel.x, actor_vel.y, actor_vel.z])

        # # Transform to lidar (Just the rotation)
        to_world = np.array(actor.get_transform().get_matrix())[:3, :3]
        to_ego = np.array(lidar_loc.get_inverse_matrix())[:3, :3]
        to_ego = np.matmul(to_ego, to_world)
        rel_vel = np.matmul(to_ego, actor_vel)

        # For checking whether velocities are correct in lidar frame
        # ego_vel = actor_list.find(int(vehicle_id)).get_velocity()
        # ego_vel = np.asarray([ego_vel.x, ego_vel.y, ego_vel.z])
        # to_world = np.array(actor_list.find(int(vehicle_id)).get_transform().get_matrix())[:3, :3]
        # to_ego = np.array(lidar_loc.get_inverse_matrix())[:3, :3]
        # to_ego = np.matmul(to_ego, to_world)
        # rel_vel = np.matmul(to_ego, ego_vel)
        # print("Ego: ", ego_vel)
        # print(rel_vel)

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


# Add autonomous vehicles to the world
def create_traffic(world, client, num_vehicle, traffic_manager):
    # Create actors
    blueprints = get_actor_blueprints(world, "vehicle.*", "All")
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


# Add autonomous people
def create_people(world, client, num_walkers):
    SpawnActor = carla.command.SpawnActor

    walkers_list = []
    all_id = []

    blueprints = get_actor_blueprints(world, "walker.pedestrian.*", '2')

    percentagePedestriansRunning = 0.0  # how many pedestrians will run
    percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road

    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprints)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if random.random() > percentagePedestriansRunning:
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2

    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id

    # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])

    all_actors = world.get_actors(all_id)

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    return all_id