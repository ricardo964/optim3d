from functools import lru_cache

import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


@lru_cache(maxsize=4)
def _ray_grid(h: int, w: int, fov: float) -> tuple[np.ndarray, np.ndarray]:
    """Rayos (u - cx) / f y (v - cy) / f para una resolución dada, cacheados."""
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return (u - cx) / f, (v - cy) / f


def depth_map_to_3d_points(depth: np.ndarray, fov: float) -> np.ndarray:
    h, w = depth.shape[:2]
    rx, ry = _ray_grid(h, w, fov)
    z = depth.astype(np.float32)
    return np.stack((rx * z, ry * z, z), axis=-1)


def valid_depth_mask(depth: np.ndarray, max_depth: float = None, edge_threshold: float = 0.05) -> np.ndarray:
    """Descarta profundidades inválidas y "flying pixels" en discontinuidades.

    Un píxel se considera borde si su gradiente relativo |∇z| / z supera edge_threshold.
    """
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        valid &= z < max_depth
    if edge_threshold is not None and edge_threshold > 0:
        gy, gx = np.gradient(z)
        valid &= np.hypot(gx, gy) < edge_threshold * np.maximum(z, 1e-6)
    return valid


def make_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return point_cloud


def voxel_cloud(point_cloud, voxel_size):
    filter_pcb = point_cloud.voxel_down_sample(voxel_size)
    filter_pcb, _ = filter_pcb.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return filter_pcb


def icp_cloud(point_cloud, voxel_size, max_points=10000):
    """Versión gruesa de la nube con normales, para registrar más rápido.

    En escenas lejanas el voxel no reduce lo suficiente, así que se limita a max_points.
    """
    coarse = point_cloud.voxel_down_sample(voxel_size)
    if max_points and len(coarse.points) > max_points:
        coarse = coarse.uniform_down_sample(int(np.ceil(len(coarse.points) / max_points)))
    coarse.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    # nube en coordenadas de cámara: la cámara está en el origen
    coarse.orient_normals_towards_camera_location(np.zeros(3))
    return coarse


def _icp(source, target, max_distance, init, max_iteration):
    return reg.registration_icp(
        source,
        target,
        max_distance,
        init,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=max_iteration),
    )


def scale_from_correspondences(source, target, result, scale_range=(0.9, 1.1)):
    """Razón mediana de distancias a la cámara de target entre puntos correspondientes.

    No se usa Sim3 punto-a-punto porque encoger la nube aumenta las correspondencias
    y la escala colapsa; la mediana de las razones no tiene ese sesgo.
    """
    corr = np.asarray(result.correspondence_set)
    if len(corr) < 100:
        return 1.0
    T = result.transformation
    src = np.asarray(source.points)[corr[:, 0]] @ T[:3, :3].T + T[:3, 3]
    dst = np.asarray(target.points)[corr[:, 1]]
    ratio = np.linalg.norm(dst, axis=1) / np.maximum(np.linalg.norm(src, axis=1), 1e-6)
    return float(np.clip(np.median(ratio), *scale_range))


def register_icp(source, target, max_distance, init=None, align_scale=False):
    """Alinea source sobre target con ICP punto-a-plano (target necesita normales).

    Con align_scale, además estima la escala de source respecto a target, escala source
    en sitio (respecto a su cámara) y refina. Devuelve (transformación 4x4 que lleva
    source al sistema de target, fitness, rmse, escala aplicada).
    """
    if init is None:
        init = np.eye(4)
    if not align_scale:
        result = _icp(source, target, max_distance, init, 50)
        return result.transformation, result.fitness, result.inlier_rmse, 1.0

    result = _icp(source, target, max_distance, init, 30)
    scale = scale_from_correspondences(source, target, result)
    # s(Rp + t) = R(sp) + st: escalar respecto a la cámara de target equivale a escalar
    # source respecto a su propia cámara y escalar la traslación
    source.scale(scale, center=np.zeros(3))
    init = result.transformation.copy()
    init[:3, 3] *= scale
    result = _icp(source, target, max_distance, init, 20)
    return result.transformation, result.fitness, result.inlier_rmse, scale
