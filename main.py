import argparse
import os
import time

import cv2
import numpy as np
import open3d as o3d
import psutil

from depth import estimate_depth
from utils import (
    depth_map_to_3d_points,
    icp_cloud,
    make_point_cloud,
    register_icp,
    valid_depth_mask,
    voxel_cloud,b
)

def merge_clouds(global_map, clouds, voxel_size):
    batch = o3d.geometry.PointCloud()
    for cloud in clouds:
        batch += cloud
    return (global_map + batch).voxel_down_sample(voxel_size)


def main(args: argparse.Namespace) -> None:
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
    process = psutil.Process()

    frame_index = 0
    previous_cloud = None
    pose = np.eye(4)  # frame actual -> sistema del primer frame
    velocity = np.eye(4)  # última transformación relativa, usada como init de ICP
    scale = 1.0  # escala acumulada de la profundidad respecto al primer frame
    global_map = o3d.geometry.PointCloud()
    pending = []  # nubes alineadas aún no fusionadas en global_map
    rejected = 0

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_depth = estimate_depth(rgb_frame, size=(work_w, work_h))
        if work_w != src_w:
            rgb_frame = cv2.resize(rgb_frame, (work_w, work_h), interpolation=cv2.INTER_AREA)

        valid = valid_depth_mask(frame_depth, args.max_depth, args.edge_threshold)
        points = depth_map_to_3d_points(frame_depth, args.fov)[valid]
        colors = rgb_frame[valid] / 255.0

        _point_cloud = voxel_cloud(make_point_cloud(points * scale, colors), args.voxel_size)
        coarse_cloud = icp_cloud(_point_cloud, args.icp_voxel_size, args.icp_max_points)

        icp_info = ""
        if previous_cloud is not None:
            transformation, fitness, rmse, frame_scale = register_icp(
                coarse_cloud, previous_cloud, args.icp_distance, velocity, args.scale_align
            )
            icp_info = f" | fitness={fitness:.3f} rmse={rmse:.4f} scale={scale * frame_scale:.3f}"

            if fitness < args.min_fitness:
                # ICP no convergió: se predice la pose con velocidad constante y el frame no se
                # integra, pero pasa a ser la referencia para no perder el seguimiento en cascada
                rejected += 1
                pose = pose @ velocity
                previous_cloud = icp_cloud(_point_cloud, args.icp_voxel_size, args.icp_max_points)
                print(f"frame {frame_index + 1}/{total_frames} | rechazado{icp_info}")
                frame_index += 1
                continue

            pose = pose @ transformation
            velocity = transformation
            _point_cloud.scale(frame_scale, center=np.zeros(3))
            # la escala vuelve lentamente a la del modelo para que no derive sin límite
            scale = (scale * frame_scale) ** (1 - args.scale_decay)

        aligned_cloud = o3d.geometry.PointCloud(_point_cloud)
        aligned_cloud.transform(pose)
        pending.append(aligned_cloud)

        # sumar al mapa copia todos sus puntos: se acumula en un buffer y se fusiona cada N frames
        if len(pending) >= args.compact_every:
            global_map = merge_clouds(global_map, pending, args.voxel_size)
            pending = []

        previous_cloud = coarse_cloud

        timestep = time.perf_counter() - cycle_start
        memory_info = f" | ram={process.memory_info().rss / 1024**2:.0f}MB"
        memory_info += f" map={len(global_map.points) + sum(len(c.points) for c in pending)}pts"

        print(f"frame {frame_index + 1}/{total_frames} | timestep={timestep:.3f}s{memory_info}{icp_info}")

        frame_index += 1

    capture.release()

    global_map = merge_clouds(global_map, pending, args.voxel_size)
    map_path = os.path.join(args.out, "map.ply")
    o3d.io.write_point_cloud(map_path, global_map)
    print(f"{map_path}: {len(global_map.points)} puntos | frames rechazados: {rejected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fov", type=float, default=60.0, help="campo de visión horizontal en grados")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--step", type=int, default=1, help="procesar uno de cada N frames")
    parser.add_argument("--width", type=int, default=640, help="ancho de trabajo para la retroproyección")
    parser.add_argument("--edge-threshold", type=float, default=0.03, help="gradiente relativo máximo |∇z|/z; 0 desactiva")
    parser.add_argument("--icp-voxel-size", type=float, default=0.03, help="voxel de la nube gruesa usada en ICP")
    parser.add_argument("--icp-max-points", type=int, default=10000, help="máximo de puntos de la nube de ICP")
    parser.add_argument("--icp-distance", type=float, default=0.05, help="distancia máxima de correspondencia para ICP")
    parser.add_argument("--min-fitness", type=float, default=0.3, help="fitness mínimo de ICP para integrar un frame")
    parser.add_argument("--no-scale-align", dest="scale_align", action="store_false", help="no corregir la escala de la profundidad entre frames")
    parser.add_argument("--scale-decay", type=float, default=0.05, help="fracción por frame con la que la escala vuelve a 1")
    parser.add_argument("--compact-every", type=int, default=20, help="fusiona y voxeliza el mapa cada N frames integrados")
    parser.add_argument("--out", default="clouds")
    args = parser.parse_args()

    main(args)
