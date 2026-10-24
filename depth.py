import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

depth_estimation_model = YOLO("yolo26n-depth.pt")

# yolo26n-depth
def estimate_depth(rgb_image: np.ndarray) -> np.ndarray:
    h, w = rgb_image.shape[:2]

    bgr_image = np.ascontiguousarray(rgb_image[..., ::-1])
    result = depth_estimation_model(bgr_image, verbose=False)[0]

    depth = result.depth.data.float().squeeze()

    if depth.shape != (h, w):
        depth = F.interpolate(depth[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]

    return depth.cpu().numpy()
