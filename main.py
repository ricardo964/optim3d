import argparse
import os
import time

import cv2
import numpy as np
import open3d as o3d
import psutil

from depth import estimate_depth
from utils import (
    clean_mesh,
    depth_scale,
    detect_features,
    estimate_pose,
    extract_surface,
    information_matrix,
    integrate_tsdf,
    intrinsics,
    optimize_pose_graph,
    reg,
    valid_depth_mask,
)

RECENT_KEYFRAMES = 2  # keyframes anteriores al nuevo que no se prueban como loop closure


class Segment:
    """Mapa conexo: keyframes enlazados por PnP, su pose graph y los frames que integra."""

    def __init__(self, features, rgb, depth):
        self.keyframes = [features]  # (xy, desc, xyz) con xyz ya en la escala del mapa
        self.centers = [np.zeros(3)]  # centro de cada keyframe, para buscar loop closures
        self.graph = reg.PoseGraph()  # nodos: pose keyframe -> mundo; aristas: T keyframe i -> j
        self.graph.nodes.append(reg.PoseGraphNode(np.eye(4)))
        self.frames = [(rgb, depth)]
        self.anchors = [(0, np.eye(4))]  # (keyframe, T keyframe -> frame) de cada frame
        self.ref = 0  # keyframe contra el que se sigue la cámara
        self.loops = 0

    def track(self, features, K, min_inliers):
        """Pose del frame respecto al keyframe de referencia; si falla, relocaliza contra el resto.

        Devuelve (índice del keyframe, T keyframe -> frame, pares inliers) o None.
        """
        result = estimate_pose(self.keyframes[self.ref], features, K, min_inliers)
        if result is not None:
            return self.ref, *result
        best = None
        for i, keyframe in enumerate(self.keyframes):
            if i == self.ref:
                continue
            result = estimate_pose(keyframe, features, K, min_inliers)
            if result is not None and (best is None or len(result[1]) > len(best[2])):
                best = (i, *result)
        return best

    def add_keyframe(self, features, pose, ref, T, pairs):
        new = len(self.keyframes)
        self.keyframes.append(features)
        self.centers.append(pose[:3, 3])
        self.graph.nodes.append(reg.PoseGraphNode(pose))
        self.graph.edges.append(reg.PoseGraphEdge(ref, new, T, self._information(ref, T, pairs), uncertain=False))
        return new

    def detect_loops(self, new, ref, K, args):
        """Añade aristas de loop closure entre el keyframe nuevo y los más cercanos a él.

        Se excluyen ref (ya enlazado por odometría) y los keyframes recientes, que siempre
        registran con el nuevo y no cierran ningún lazo.
        """
        distances = np.linalg.norm(np.array(self.centers[:new]) - self.centers[new], axis=1)
        candidates = [i for i in np.argsort(distances) if i != ref and i < new - RECENT_KEYFRAMES]
        candidates = candidates[: args.loop_candidates]
        loops = 0
        for i in candidates:
            result = estimate_pose(self.keyframes[i], self.keyframes[new], K, args.loop_inliers)
            if result is None:
                continue
            T, pairs = result
            self.graph.edges.append(reg.PoseGraphEdge(i, new, T, self._information(i, T, pairs), uncertain=True))
            loops += 1
        self.loops += loops
        return loops

    def _information(self, i, T, pairs):
        return information_matrix(self.keyframes[i][2][pairs[:, 0]] @ T[:3, :3].T + T[:3, 3])

    def poses(self, loop_distance):
        if self.loops:
            optimize_pose_graph(self.graph, loop_distance)
        return [self.graph.nodes[kf].pose @ np.linalg.inv(T) for kf, T in self.anchors]


def main(args: argparse.Namespace) -> None:
    cv2.setRNGSeed(0)  # RANSAC reproducible entre ejecuciones
    os.makedirs(args.out, exist_ok=True)

    # mp4 to frame
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # el FOV es horizontal, así que reducir la resolución mantiene los intrínsecos coherentes
    work_w = min(args.width, src_w)
    work_h = round(src_h * work_w / src_w)
    K = intrinsics(work_w, work_h, args.fov)
    process = psutil.Process()

    segments = []
    segment = None
    lost = None  # último frame sin pose: semilla de un segmento nuevo si el siguiente registra con él
    processed = 0
    frame_index = 0

    try:
        while True:
            if frame_index % args.step != 0:
                if not capture.grab():
                    break
                frame_index += 1
                continue

            ok, frame = capture.read()
            if not ok:
                break

            cycle_start = time.perf_counter()
            processed += 1

            depth = estimate_depth(frame, size=(work_w, work_h))
            depth[~valid_depth_mask(depth, args.max_depth, args.depth_percentile, args.edge_threshold)] = 0
            features = detect_features(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), depth, K)
            if work_w != src_w:
                frame = cv2.resize(frame, (work_w, work_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            info = ""
            if segment is None:
                segment = Segment(features, rgb, depth)
                segments.append(segment)
                info = " | segmento 0"
            else:
                tracked = segment.track(features, K, args.min_inliers)
                if tracked is None and lost is not None:
                    seed = estimate_pose(lost[0], features, K, args.min_inliers)
                    if seed is not None:
                        # dos frames seguidos fuera del mapa pero coherentes entre sí: la cámara
                        # llegó a una zona sin solape y se empieza un mapa nuevo desde el perdido
                        segment = Segment(*lost)
                        segments.append(segment)
                        tracked = (0, *seed)
                        info = f" | segmento {len(segments) - 1} desde el frame anterior"

                if tracked is None:
                    # sin pose verificada el frame deformaría el mapa: no se integra
                    lost = (features, rgb, depth)
                    print(f"frame {frame_index + 1}/{total_frames} | rechazado: sin pose")
                    frame_index += 1
                    continue

                lost = None
                ref, T, pairs = tracked
                # la pose se calculó con la escala del keyframe: se lleva este frame a esa escala
                # y se relaja hacia la del modelo para que la escala no derive sin límite
                scale = depth_scale(T, segment.keyframes[ref][2], features[2], pairs) ** (1 - args.scale_decay)
                depth *= scale
                features[2][:] *= scale
                pose = segment.graph.nodes[ref].pose @ np.linalg.inv(T)
                info += f" | kf={ref} inliers={len(pairs)} scale={scale:.3f}"

                if len(pairs) < args.keyframe_inliers:
                    new = segment.add_keyframe(features, pose, ref, T, pairs)
                    info += f" | keyframe {new} loops={segment.detect_loops(new, ref, K, args)}"
                    segment.ref, anchor = new, (new, np.eye(4))
                else:
                    segment.ref, anchor = ref, (ref, T)
                segment.frames.append((rgb, depth))
                segment.anchors.append(anchor)

            timestep = time.perf_counter() - cycle_start
            memory_info = f" | ram={process.memory_info().rss / 1024**2:.0f}MB"
            print(f"frame {frame_index + 1}/{total_frames} | timestep={timestep:.3f}s{memory_info}{info}")

            frame_index += 1
    finally:
        capture.release()

    if not segments:
        raise RuntimeError(f"No se pudo leer ningún frame de {args.video}")
    rejected = processed - sum(len(s.frames) for s in segments)

    # los segmentos no comparten sistema de referencia: se exporta el que cubre más frames
    best = max(segments, key=lambda s: len(s.frames))
    volume = integrate_tsdf(best.frames, best.poses(args.loop_distance), K, args.voxel_size, args.sdf_trunc)
    global_map, mesh = extract_surface(volume, args.min_weight)
    if not global_map.has_points():
        raise RuntimeError("El TSDF no tiene superficie: ningún frame aportó profundidad válida")
    mesh = clean_mesh(mesh, args.min_component, args.smooth_iterations)
    map_path = os.path.join(args.out, "map.ply")
    mesh_path = os.path.join(args.out, "mesh.ply")
    o3d.io.write_point_cloud(map_path, global_map)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(
        f"{map_path}: {len(global_map.points)} puntos | {mesh_path}: {len(mesh.triangles)} triángulos "
        f"| segmentos (frames): {[len(s.frames) for s in segments]} | keyframes: {len(best.keyframes)} "
        f"| loop closures: {best.loops} | rechazados: {rejected}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fov", type=float, default=60.0, help="campo de visión horizontal en grados")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="voxel del volumen TSDF")
    parser.add_argument("--sdf-trunc", type=float, default=0.1, help="distancia de truncado del TSDF; ~10 voxels absorbe el ruido de la profundidad monocular")
    parser.add_argument("--min-weight", type=float, default=3, help="peso TSDF mínimo para extraer superficie; subirlo exige que la vean varios frames")
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--depth-percentile", type=float, default=95, help="descarta el fondo por encima de este percentil de profundidad; 100 desactiva")
    parser.add_argument("--step", type=int, default=1, help="procesar uno de cada N frames")
    parser.add_argument("--width", type=int, default=640, help="ancho de trabajo para la retroproyección")
    parser.add_argument("--edge-threshold", type=float, default=0.03, help="gradiente relativo máximo |∇z|/z; 0 desactiva")
    parser.add_argument("--min-inliers", type=int, default=40, help="inliers PnP mínimos para aceptar la pose de un frame")
    parser.add_argument("--keyframe-inliers", type=int, default=200, help="crea un keyframe cuando los inliers contra el actual bajan de este valor")
    parser.add_argument("--loop-candidates", type=int, default=5, help="keyframes más cercanos probados como loop closure")
    parser.add_argument("--loop-inliers", type=int, default=60, help="inliers PnP mínimos para aceptar un loop closure")
    parser.add_argument("--loop-distance", type=float, default=0.05, help="residuo de referencia del pose graph para podar loop closures")
    parser.add_argument("--scale-decay", type=float, default=0.05, help="fracción por frame con la que la escala vuelve a 1")
    parser.add_argument("--min-component", type=float, default=0.01, help="borra de la malla las islas con menos de esta fracción de triángulos; 0 desactiva")
    parser.add_argument("--smooth-iterations", type=int, default=10, help="iteraciones de suavizado Taubin de la malla; 0 desactiva")
    parser.add_argument("--out", default="clouds")
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step debe ser >= 1")
    if min(args.min_inliers, args.loop_inliers) < 4:
        parser.error("PnP necesita al menos 4 inliers: --min-inliers y --loop-inliers deben ser >= 4")

    main(args)
