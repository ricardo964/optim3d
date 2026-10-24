import numpy as np
import open3d as o3d

def depth_map_to_3d_points(image, fov) -> np.array:
    h, w = image.shape[:2]
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    cx, cy = (w - 1) / 2, (h - 1) / 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = image.astype(np.float32)
    x = (u - cx) * z / f
    y = (v - cy) * z / f

    return np.stack((x, y, z), axis=-1)

def points_map_to_cloud(points_map: np.ndarray, max_depth: float = None) -> np.ndarray:
    cloud = points_map.reshape(-1, 3)
    if max_depth is not None:
        cloud = cloud[cloud[:, 2] < max_depth]
    return cloud

def voxel_cloud(point_cloud, voxel_size):
    filter_pcb = point_cloud.voxel_down_sample(voxel_size)
    filter_pcb, ind = filter_pcb.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    filter_pcb.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    return filter_pcb

