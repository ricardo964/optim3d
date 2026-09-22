import queue
import random
from dataclasses import dataclass, field
from pathlib import Path

import carla
import pandas as pd


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fov: str = "90"
    x: float = 1.5
    y: float = 0.0
    z: float = 2.0


@dataclass(frozen=True)
class SimulationConfig:
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    fixed_delta_seconds: float = 0.05
    tm_port: int = 8000
    max_frames: int = 10000
    output_dir: Path = Path("output")
    speed_diff_percentage: float = -20.0
    distance_to_leading_vehicle: float = 2.0
    ignore_lights_percentage: float = 0.0
    vehicle_blueprint_id: str = "vehicle.tesla.model3"
    camera: CameraConfig = field(default_factory=CameraConfig)


class SynchronousModeContext:
    def __init__(self, world, traffic_manager, fixed_delta_seconds):
        self._world = world
        self._tm = traffic_manager
        self._fixed_delta_seconds = fixed_delta_seconds
        self._original_settings = None

    def __enter__(self):
        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._fixed_delta_seconds
        self._world.apply_settings(settings)
        self._tm.set_synchronous_mode(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tm.set_synchronous_mode(False)
        self._world.apply_settings(self._original_settings)
        return False


class VehicleFactory:
    def __init__(self, world, blueprint_id):
        self._world = world
        self._blueprint_id = blueprint_id

    def spawn(self):
        blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self._blueprint_id)[0]
        spawn_points = self._world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = self._world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            raise RuntimeError(f"No se pudo spawnear el vehiculo en {spawn_point}")
        return vehicle


class AutopilotController:
    def __init__(self, traffic_manager, tm_port, config: SimulationConfig):
        self._tm = traffic_manager
        self._tm_port = tm_port
        self._config = config

    def enable(self, vehicle):
        vehicle.set_autopilot(True, self._tm_port)
        self._tm.ignore_lights_percentage(vehicle, self._config.ignore_lights_percentage)
        self._tm.vehicle_percentage_speed_difference(vehicle, self._config.speed_diff_percentage)
        self._tm.distance_to_leading_vehicle(vehicle, self._config.distance_to_leading_vehicle)
        self._tm.set_random_device_seed(random.randint(0, 10000))


class SensorFactory:
    def __init__(self, world, config: CameraConfig):
        self._world = world
        self._config = config

    def _build_transform(self):
        return carla.Transform(
            carla.Location(x=self._config.x, y=self._config.y, z=self._config.z),
            carla.Rotation(pitch=0, yaw=0, roll=0),
        )

    def _configure_blueprint(self, blueprint_id):
        blueprint_library = self._world.get_blueprint_library()
        bp = blueprint_library.find(blueprint_id)
        bp.set_attribute("image_size_x", str(self._config.width))
        bp.set_attribute("image_size_y", str(self._config.height))
        bp.set_attribute("fov", self._config.fov)
        return bp

    def create_rgb_camera(self, vehicle):
        bp = self._configure_blueprint("sensor.camera.rgb")
        return self._world.spawn_actor(bp, self._build_transform(), attach_to=vehicle)

    def create_depth_camera(self, vehicle):
        bp = self._configure_blueprint("sensor.camera.depth")
        return self._world.spawn_actor(bp, self._build_transform(), attach_to=vehicle)


class SensorStream:
    def __init__(self, sensor):
        self._sensor = sensor
        self._queue = queue.Queue()
        self._sensor.listen(self._queue.put)

    def get(self):
        return self._queue.get()

    def stop(self):
        self._sensor.stop()

    def destroy(self):
        self._sensor.destroy()


class DatasetRecorder:
    _COLUMNS = ["x", "y", "yaw", "image_filename"]

    def __init__(self, output_dir: Path):
        self._rgb_dir = output_dir / "rgb"
        self._depth_dir = output_dir / "depth"
        self._rgb_dir.mkdir(parents=True, exist_ok=True)
        self._depth_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = output_dir / "metric.csv"
        pd.DataFrame(columns=self._COLUMNS).to_csv(self._csv_path, index=False)

    def record(self, frame_index, vehicle, rgb_image, depth_image):
        filename = f"{frame_index:06d}.png"
        transform = vehicle.get_transform()
        location = transform.location
        yaw = transform.rotation.yaw

        row = pd.DataFrame([[location.x, location.y, yaw, filename]], columns=self._COLUMNS)
        row.to_csv(self._csv_path, mode="a", header=False, index=False)

        rgb_image.save_to_disk(str(self._rgb_dir / filename))
        depth_image.save_to_disk(str(self._depth_dir / filename), carla.ColorConverter.LogarithmicDepth)

    def close(self):
        pass


class DatasetCollector:
    def __init__(self, config: SimulationConfig):
        self._config = config
        self._client = carla.Client(config.host, config.port)
        self._client.set_timeout(config.timeout)
        self._world = self._client.get_world()
        self._tm = self._client.get_trafficmanager(config.tm_port)

    def run(self):
        print("Mapa:", self._world.get_map().name)

        vehicle = VehicleFactory(self._world, self._config.vehicle_blueprint_id).spawn()
        print("Vehiculo:", vehicle.id)

        AutopilotController(self._tm, self._tm.get_port(), self._config).enable(vehicle)

        sensor_factory = SensorFactory(self._world, self._config.camera)
        rgb_stream = SensorStream(sensor_factory.create_rgb_camera(vehicle))
        depth_stream = SensorStream(sensor_factory.create_depth_camera(vehicle))

        recorder = DatasetRecorder(self._config.output_dir)

        try:
            with SynchronousModeContext(self._world, self._tm, self._config.fixed_delta_seconds):
                for frame_index in range(self._config.max_frames):
                    self._world.tick()
                    rgb_image = rgb_stream.get()
                    depth_image = depth_stream.get()
                    recorder.record(frame_index, vehicle, rgb_image, depth_image)

                    if (frame_index + 1) % 100 == 0:
                        print(f"Frames capturados: {frame_index + 1}")
        except KeyboardInterrupt:
            print("Interrumpido por el usuario")
        finally:
            print("Limpiando actores...")
            rgb_stream.stop()
            depth_stream.stop()
            rgb_stream.destroy()
            depth_stream.destroy()
            vehicle.destroy()
            recorder.close()
            print("Listo.")


def main():
    config = SimulationConfig(max_frames=2000 * 5)
    DatasetCollector(config).run()


if __name__ == "__main__":
    main()