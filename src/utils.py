import numpy as np

def carla_depth_map_to_3d_points(image, fov) -> np.array:
    B = image[:, :, 0].astype(np.float64)
    G = image[:, :, 1].astype(np.float64)
    R = image[:, :, 2].astype(np.float64)

    normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000.0

    height, width = image.shape[:2]

    f = width / (2.0 * np.tan(fov * np.pi / (2.0 * 180.0)))
    cx = width / 2.0
    cy = height / 2.0

    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)

    Z = depth_meters
    X = (uu - cx) * Z / f
    Y = (vv - cy) * Z / f

    points_3d = np.stack([X, Y, Z], axis=-1)
    return points_3d


def points_map_to_cloud(points_map: np.ndarray, max_depth: float = None) -> np.ndarray:
    cloud = points_map.reshape(-1, 3)
    if max_depth is not None:
        cloud = cloud[cloud[:, 2] < max_depth]
    return cloud

