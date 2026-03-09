from __future__ import annotations

from typing import Optional, Sequence, Tuple
import math
import numpy as np


def normalize_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi



def rotation_matrix_z(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)



def make_transform(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T



def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    hom = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1), dtype=float)], axis=1)
    out = (np.asarray(T, dtype=float) @ hom.T).T
    return out[:, :3]



def point_sensor_to_ego(p_sensor: Sequence[float], sensor_to_ego: np.ndarray) -> np.ndarray:
    return transform_points(np.asarray(p_sensor, dtype=float), sensor_to_ego)[0]



def vector_sensor_to_ego(v_sensor: Sequence[float], sensor_to_ego: np.ndarray) -> np.ndarray:
    return np.asarray(sensor_to_ego, dtype=float)[:3, :3] @ np.asarray(v_sensor, dtype=float)



def corners_from_box_ego(center: Sequence[float], size: Sequence[float], yaw: float) -> np.ndarray:
    cx, cy, cz = [float(v) for v in center]
    l, w, h = [float(v) for v in size]
    x = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=float)
    y = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=float)
    z = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dtype=float)
    corners = np.stack([x, y, z], axis=1)
    R = rotation_matrix_z(yaw)
    corners = (R @ corners.T).T
    corners += np.array([cx, cy, cz], dtype=float)
    return corners



def project_points_to_image(points_ego: np.ndarray, ego_to_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts_cam = transform_points(points_ego, ego_to_cam)
    depths = pts_cam[:, 2].copy()
    valid = depths > 1e-3
    uv = np.full((pts_cam.shape[0], 2), np.nan, dtype=float)
    if np.any(valid):
        proj = (np.asarray(K, dtype=float) @ pts_cam[valid].T).T
        uv_valid = proj[:, :2] / proj[:, 2:3]
        uv[valid] = uv_valid
    return uv, valid



def box_to_image_bbox(center_ego: Sequence[float], size: Sequence[float], yaw: float,
                      ego_to_cam: np.ndarray, K: np.ndarray,
                      image_size: Optional[Tuple[int, int]] = None) -> Optional[Tuple[float, float, float, float]]:
    corners = corners_from_box_ego(center_ego, size, yaw)
    uv, valid = project_points_to_image(corners, ego_to_cam, K)
    if not np.any(valid):
        return None
    uv_valid = uv[valid]
    x1, y1 = np.nanmin(uv_valid[:, 0]), np.nanmin(uv_valid[:, 1])
    x2, y2 = np.nanmax(uv_valid[:, 0]), np.nanmax(uv_valid[:, 1])
    if image_size is not None:
        w, h = image_size
        x1 = float(np.clip(x1, 0, w - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        y2 = float(np.clip(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return float(x1), float(y1), float(x2), float(y2)



def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)



def mahalanobis_or_euclidean(p: Sequence[float], q: Sequence[float], cov: Optional[np.ndarray]) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    dp = (p[:2] - q[:2]).reshape(2, 1)
    if cov is None:
        return float(np.linalg.norm(dp[:, 0]))
    try:
        cov2 = np.asarray(cov, dtype=float)[:2, :2] + 1e-6 * np.eye(2)
        inv = np.linalg.inv(cov2)
        val = float((dp.T @ inv @ dp).item())
        return float(math.sqrt(max(0.0, val)))
    except Exception:
        return float(np.linalg.norm(dp[:, 0]))



def circular_mean(angles: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    if len(angles) == 0:
        return 0.0
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        weights = np.ones(len(angles), dtype=float)
    weights = np.asarray(weights, dtype=float)
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return normalize_angle(math.atan2(s, c))



def yaw_from_rotation_matrix(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=float)
    return normalize_angle(math.atan2(R[1, 0], R[0, 0]))



def global_to_ego_xy(points_global: np.ndarray, ego_translation_global: Sequence[float], ego_yaw_global: float) -> np.ndarray:
    pts = np.asarray(points_global, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    translated = pts[:, :2] - np.asarray(ego_translation_global, dtype=float)[:2]
    c = math.cos(-float(ego_yaw_global))
    s = math.sin(-float(ego_yaw_global))
    R = np.array([[c, -s], [s, c]], dtype=float)
    return (R @ translated.T).T



def relative_direction_label(pos_xy: Sequence[float]) -> str:
    x, y = float(pos_xy[0]), float(pos_xy[1])
    if abs(y) < 1.5:
        return "front" if x >= 0 else "behind"
    if x >= 0 and y > 0:
        return "front_left"
    if x >= 0 and y < 0:
        return "front_right"
    if x < 0 and y > 0:
        return "rear_left"
    return "rear_right"
