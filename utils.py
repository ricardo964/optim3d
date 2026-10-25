import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

reg = o3d.pipelines.registration

sift = cv2.SIFT_create(nfeatures=2000)
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def intrinsics(w: int, h: int, fov: float) -> np.ndarray:
    """Matriz K pinhole para un FOV horizontal en grados, con el centro óptico en el centro."""
    f = (w / 2) / np.tan(np.radians(fov) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])


def valid_depth_mask(depth, max_depth=None, percentile=None, edge_threshold=0.03) -> np.ndarray:
    """Descarta profundidades inválidas, el fondo lejano y "flying pixels" en discontinuidades.

    percentile recorta el fondo de cada frame (cielo, ventanas), donde la profundidad
    monocular es menos fiable. Un píxel es borde si su gradiente relativo |∇z| / z supera
    edge_threshold.
    """
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        valid &= z < max_depth
    if percentile is not None and percentile < 100 and valid.any():
        valid &= z <= np.percentile(z[valid], percentile)
    if edge_threshold:
        gy, gx = np.gradient(z)
        valid &= np.hypot(gx, gy) < edge_threshold * np.maximum(z, 1e-6)
    return valid


def detect_features(gray: np.ndarray, depth: np.ndarray, K: np.ndarray):
    """Keypoints SIFT (xy), descriptores y su posición 3D en cámara (NaN sin profundidad válida).

    gray puede tener más resolución que depth: SIFT encuentra muchos más keypoints en el
    frame original, y sus coordenadas se llevan a la resolución de trabajo de depth y K.
    """
    keypoints, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32), np.empty((0, 3))
    h, w = depth.shape
    # cambio de resolución por eje respetando el centro de píxel: x' = (x + 0.5) * s - 0.5
    factor = np.float32([w / gray.shape[1], h / gray.shape[0]])
    xy = (np.float32([k.pt for k in keypoints]) + 0.5) * factor - 0.5
    u = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    z = depth[v, u].astype(np.float64)
    xyz = np.column_stack(((xy[:, 0] - K[0, 2]) / K[0, 0] * z, (xy[:, 1] - K[1, 2]) / K[1, 1] * z, z))
    xyz[z <= 0] = np.nan
    return xy, desc, xyz


def match_features(desc_a, desc_b, ratio=0.75) -> np.ndarray:
    """Pares (i_a, i_b) que pasan el ratio test de Lowe."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), int)
    pairs = []
    for knn in matcher.knnMatch(desc_a, desc_b, k=2):
        if len(knn) == 2 and knn[0].distance < ratio * knn[1].distance:
            pairs.append((knn[0].queryIdx, knn[0].trainIdx))
    return np.array(pairs, int).reshape(-1, 2)


def estimate_pose(src, dst, K, min_inliers, reprojection_error=6.0):
    """PnP + RANSAC entre los puntos 3D de src y los keypoints 2D de dst.

    src y dst son (xy, desc, xyz). Devuelve (T 4x4 que lleva puntos de la cámara src a la
    cámara dst, pares inliers) o None si no hay suficientes inliers.
    """
    pairs = match_features(src[1], dst[1])
    pairs = pairs[np.isfinite(src[2][pairs[:, 0], 2])]
    if len(pairs) < min_inliers:
        return None
    obj = src[2][pairs[:, 0]]
    img = dst[0][pairs[:, 1]].astype(np.float64)
    found, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, None, iterationsCount=500, reprojectionError=reprojection_error, confidence=0.999
    )
    if not found or inliers is None or len(inliers) < min_inliers:
        return None
    inliers = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inliers], img[inliers], K, None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, pairs[inliers]


def depth_scale(T, src_xyz, dst_xyz, pairs, scale_range=(0.5, 2.0)) -> float:
    """Escala de la profundidad de dst respecto a src: z que predice la pose / z de la red."""
    z_pred = (src_xyz[pairs[:, 0]] @ T[:3, :3].T + T[:3, 3])[:, 2]
    z = dst_xyz[pairs[:, 1], 2]
    ok = np.isfinite(z) & (z_pred > 0)
    if ok.sum() < 20:
        return 1.0
    return float(np.clip(np.median(z_pred[ok] / z[ok]), *scale_range))


def information_matrix(points: np.ndarray) -> np.ndarray:
    """Matriz de información 6x6 de un registro, como get_information_matrix_from_point_clouds
    de Open3D pero a partir de los inliers (en coordenadas de la cámara destino)."""
    x, y, z = points.T
    G = np.zeros((len(points), 3, 6))
    G[:, 0, 1], G[:, 0, 2], G[:, 0, 3] = z, -y, 1
    G[:, 1, 0], G[:, 1, 2], G[:, 1, 4] = -z, x, 1
    G[:, 2, 0], G[:, 2, 1], G[:, 2, 5] = y, -x, 1
    return np.einsum("nki,nkj->ij", G, G)


def optimize_pose_graph(graph, max_distance):
    """Optimización global con line process: descarta los loop closures inconsistentes."""
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=max_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=0.1,
        reference_node=0,
    )
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        reg.global_optimization(
            graph,
            reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            option,
        )


def integrate_tsdf(frames, poses, K, voxel_size, sdf_trunc):
    """Fusiona frames (rgb uint8, profundidad en metros) con sus poses cámara -> mundo.

    Se usa VoxelBlockGrid en CPU: en Open3D 0.20 ScalableTSDFVolume devuelve volúmenes
    vacíos y VoxelBlockGrid en CUDA se bloquea junto a torch.
    """
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=(1, 1, 3),
        voxel_size=voxel_size,
        block_resolution=8,
        block_count=1000,  # el hashmap crece según haga falta
    )
    intrinsic = o3c.Tensor(K, o3c.float64)
    trunc = sdf_trunc / voxel_size
    depth_max = 1e6  # el fondo ya se recortó en valid_depth_mask
    for (rgb, depth), pose in zip(frames, poses):
        if not (depth > 0).any():
            continue  # Open3D aborta la integración si el frame no toca ningún bloque
        depth = o3d.t.geometry.Image(o3c.Tensor(depth.astype(np.float32)))
        color = o3d.t.geometry.Image(o3c.Tensor(rgb.astype(np.float32) / 255))
        extrinsic = o3c.Tensor(np.linalg.inv(pose), o3c.float64)
        blocks = volume.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, 1.0, depth_max, trunc_voxel_multiplier=trunc
        )
        volume.integrate(
            blocks, depth, color, intrinsic, intrinsic, extrinsic, 1.0, depth_max, trunc_voxel_multiplier=trunc
        )
    return volume


def extract_surface(volume, min_weight):
    """Nube y malla de la superficie del TSDF; el color interpolado puede salir de [0, 1]."""
    cloud = volume.extract_point_cloud(weight_threshold=min_weight)
    mesh = volume.extract_triangle_mesh(weight_threshold=min_weight)
    cloud.point.colors = cloud.point.colors.clip(0, 1)
    mesh.vertex.colors = mesh.vertex.colors.clip(0, 1)
    return cloud.to_legacy(), mesh.to_legacy()


def clean_mesh(mesh, min_component=0.01, smooth_iterations=10):
    """Quita las islas con menos de min_component de los triángulos y suaviza con Taubin,
    que a diferencia del laplaciano no encoge la malla."""
    if min_component > 0 and mesh.has_triangles():
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters, counts = np.asarray(clusters), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[clusters] < min_component * len(mesh.triangles))
        mesh.remove_unreferenced_vertices()
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        # Taubin también suaviza el color y puede sacarlo de [0, 1]
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(mesh.vertex_colors), 0, 1))
    mesh.compute_vertex_normals()
    return mesh
