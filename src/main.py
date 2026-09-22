import cv2
from utils import *
import argparse
import os
import open3d as o3d

def main(args: dict[str, any]) -> None:
    directory = args["directory"]

    for image_path in os.listdir(directory):
        path = os.path.join(directory, image_path)
        image = cv2.imread(path)

        position_image = carla_depth_map_to_3d_points(image, 80)
        points = points_map_to_cloud(position_image)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud("frame.ply", point_cloud)

        break # test only


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    main(vars(args))
