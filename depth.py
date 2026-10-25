import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# FP16 (half/quantize) produce NaN con este modelo: se mantiene FP32
depth_estimation_model = YOLO("yolo26n-depth.pt")


# yolo26n-depth
def estimate_depth(bgr_image: np.ndarray, size: tuple[int, int] = None) -> np.ndarray:
    """Profundidad del frame BGR; si se da size=(w, h) se redimensiona en GPU a ese tamaño."""
    h, w = bgr_image.shape[:2]
    out_w, out_h = size if size is not None else (w, h)

    result = depth_estimation_model(bgr_image, device=DEVICE, verbose=False)[0]

    depth = result.depth.data.float().squeeze()

    if depth.shape != (out_h, out_w):
        depth = F.interpolate(depth[None, None], size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]

    return depth.cpu().numpy()
