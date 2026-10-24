import cv2
from utils import *
from depth import *
import argparse
import os
import open3d as o3d

def main(args: dict[str, any]) -> None:
    video = args["video"]
    os.makedirs(args["out"], exist_ok=True)

    # mp4 to frame
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video}")

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % args["step"] != 0:
            frame_index += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_depth = estimate_depth(rgb_frame)
        points = depth_map_to_3d_points(frame_depth, args["fov"])

        if args["max_depth"] is not None:
            valid = points[..., 2] < args["max_depth"]
        else:
            valid = np.ones(points.shape[:2], dtype=bool)

        cloud = points_map_to_cloud(points, args["max_depth"])
        colors = rgb_frame[valid].reshape(-1, 3) / 255.0

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cloud)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        _point_cloud = voxel_cloud(point_cloud, args["voxel_size"])

        
        
        # path = os.path.join(args["out"], f"frame_{frame_index:05d}.ply")
        # o3d.io.write_point_cloud(path, _point_cloud)
        # print(f"{path}: {len(point_cloud.points)} -> {len(_point_cloud.points)} puntos")

        frame_index += 1
        break

    capture.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fov", type=float, default=60.0, help="campo de visión horizontal en grados")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--step", type=int, default=1, help="procesar uno de cada N frames")
    parser.add_argument("--out", default="clouds")
    args = parser.parse_args()

    main(vars(args))
