import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from CarlaUtils import *


def main(arg):
    """Main function of the script"""
    client = carla.Client(arg["host"], arg["port"])
    client.set_timeout(5.0)
    world = client.get_world()
    print("Available worlds:", client.get_available_maps())
    if arg["new_world"]:
        world = client.load_world(arg["new_world"])

    if not os.path.exists(arg["storage_dir"]):
        os.makedirs(arg["storage_dir"])

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
        walker_id = create_people(world, client, arg["num_walkers"])

        # Create semantic lidar
        NUM_SENSORS = arg["num_sensors"]
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
                offsets = np.random.uniform([-20, -20, 1], [20, 20, 5], [3,])

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
        while frame < 2000:
            world.tick()
            spectator.set_transform(cam01.get_transform())
            time.sleep(0.005)
            # Store Data
            try:
                for _ in range(len(lidars)):
                    data, view = lidar_queue.get()
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

        for lidar in lidars:
            lidar.destroy()
        for camera in cameras:
            camera.destroy()

        vehicle.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        all_actors = world.get_actors(walker_id)
        for i in range(0, len(walker_id), 2):
            all_actors[i].stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in walker_id])


if __name__ == "__main__":
    args = {
        "host": "localhost",
        "port": 2000,
        "no_rendering": False,
        "noise": False,
        "filter": "model3",
        "upper_fov": 2,
        "lower_fov": -25,
        "channels": 64.0,
        "range": 50,
        "points_per_second": 1300000,
        "show_axis": True,
        "storage_dir": "/home/tigeriv/Data/Carla/Data/Scenes/Town07_Light/raw",
        "num_vehicles": 25,
        "num_walkers": 25,
        "new_world": "/Game/Carla/Maps/Town07",
        "num_sensors": 20
    }

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')
