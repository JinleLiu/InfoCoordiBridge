from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import ICAConfig
from .geometry import bbox_iou, box_to_image_bbox, mahalanobis_or_euclidean
from .schemas import Observation


@dataclass
class EntitySeed:
    seed_id: str
    observations: List[Observation] = field(default_factory=list)

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    def modalities(self) -> List[str]:
        return sorted(set(obs.modality for obs in self.observations))

    def geometry_observations(self) -> List[Observation]:
        return [o for o in self.observations if o.position is not None]

    def best_geometry(self) -> Optional[Observation]:
        geo = self.geometry_observations()
        if not geo:
            return None
        priority = {"lidar": 4, "bevfusion": 3, "radar": 2, "camera": 1}
        return max(geo, key=lambda o: (priority.get(o.modality, 0), o.score))

    def current_state_for_projection(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        best = self.best_geometry()
        if best is None or best.position is None or best.size is None:
            return None
        yaw = 0.0 if best.yaw is None else float(best.yaw)
        return np.asarray(best.position, dtype=float), np.asarray(best.size, dtype=float), yaw



def _class_compatible_name(a: str, b: str) -> bool:
    a = a.lower()
    b = b.lower()
    if a == "unknown" or b == "unknown" or a == b:
        return True
    aliases = [
        {"car", "van", "suv", "mpv"},
        {"bus", "truck", "construction_vehicle", "trailer"},
        {"pedestrian", "adult", "child", "human"},
        {"bicycle", "cyclist"},
        {"motorcycle", "motorcyclist"},
    ]
    return any(a in s and b in s for s in aliases)



def _class_compatible(a: Observation, b: Observation) -> bool:
    return _class_compatible_name(a.class_name, b.class_name)



def _size_distance(a: Observation, b: Observation) -> float:
    if a.size is None or b.size is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(a.size) - np.asarray(b.size)))



def geometry_match_cost(a: Observation, b: Observation, cfg: ICAConfig, extra_velocity_cost: bool = False) -> float:
    if a.position is None or b.position is None:
        return float("inf")
    cov = a.covariance if a.covariance is not None else b.covariance
    d = mahalanobis_or_euclidean(a.position, b.position, cov)
    if d > cfg.matching_distance_gate_m:
        return float("inf")
    cost = d
    if not _class_compatible(a, b):
        cost += cfg.class_mismatch_penalty
    cost += cfg.size_penalty_weight * _size_distance(a, b)
    if extra_velocity_cost and a.velocity is not None and b.velocity is not None:
        cost += cfg.velocity_penalty_weight * float(np.linalg.norm(np.asarray(a.velocity)[:2] - np.asarray(b.velocity)[:2]))
    return float(cost)



def camera_attachment_cost(seed: EntitySeed, cam_det: Observation, cfg: ICAConfig) -> float:
    if cam_det.bbox_2d is None or cam_det.camera_calib is None:
        return float("inf")
    state = seed.current_state_for_projection()
    if state is None:
        return float("inf")
    center, size, yaw = state
    ego_to_cam = np.linalg.inv(np.asarray(cam_det.camera_calib.sensor_to_ego, dtype=float))
    pred_bbox = box_to_image_bbox(center, size, yaw, ego_to_cam, cam_det.camera_calib.intrinsics, cam_det.camera_calib.image_size)
    if pred_bbox is None:
        return float("inf")
    iou = bbox_iou(pred_bbox, cam_det.bbox_2d)
    if iou < cfg.camera_iou_min:
        return float("inf")
    best = seed.best_geometry()
    class_penalty = 0.0
    if best is not None and not _class_compatible(best, cam_det):
        class_penalty = cfg.camera_cost_class_penalty
    return float((1.0 - iou) + class_penalty)



def _hungarian_assign(cost_matrix: np.ndarray, inf_cost: float = 1e6) -> List[Tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    work = np.array(cost_matrix, dtype=float)
    work[~np.isfinite(work)] = inf_cost
    rows, cols = linear_sum_assignment(work)
    matches = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if work[i, j] >= inf_cost:
            continue
        matches.append((i, j))
    return matches



def stage1_build_seeds(lidar_obs: List[Observation], bev_obs: List[Observation], cfg: ICAConfig) -> List[EntitySeed]:
    if not lidar_obs and not bev_obs:
        return []
    cost = np.full((len(lidar_obs), len(bev_obs)), float("inf"), dtype=float)
    for i, l in enumerate(lidar_obs):
        for j, b in enumerate(bev_obs):
            cost[i, j] = geometry_match_cost(l, b, cfg, extra_velocity_cost=False)
    matches = _hungarian_assign(cost)
    used_l, used_b = set(), set()
    seeds: List[EntitySeed] = []
    idx = 0
    for i, j in matches:
        used_l.add(i)
        used_b.add(j)
        seed = EntitySeed(seed_id=f"seed_{idx:04d}")
        seed.add(lidar_obs[i])
        seed.add(bev_obs[j])
        seeds.append(seed)
        idx += 1
    for i, obs in enumerate(lidar_obs):
        if i not in used_l:
            seeds.append(EntitySeed(seed_id=f"seed_{idx:04d}", observations=[obs]))
            idx += 1
    for j, obs in enumerate(bev_obs):
        if j not in used_b:
            seeds.append(EntitySeed(seed_id=f"seed_{idx:04d}", observations=[obs]))
            idx += 1
    return seeds



def stage2_attach_radar(seeds: List[EntitySeed], radar_obs: List[Observation], cfg: ICAConfig) -> Tuple[List[EntitySeed], List[Observation]]:
    if not seeds or not radar_obs:
        return seeds, radar_obs
    cost = np.full((len(radar_obs), len(seeds)), float("inf"), dtype=float)
    for i, r in enumerate(radar_obs):
        for j, seed in enumerate(seeds):
            best = seed.best_geometry()
            if best is None:
                continue
            cost[i, j] = geometry_match_cost(r, best, cfg, extra_velocity_cost=True)
    matches = _hungarian_assign(cost)
    used_r = set()
    for i, j in matches:
        seeds[j].add(radar_obs[i])
        used_r.add(i)
    unmatched = [obs for i, obs in enumerate(radar_obs) if i not in used_r]
    return seeds, unmatched



def stage3_attach_camera(seeds: List[EntitySeed], cam_obs: List[Observation], cfg: ICAConfig) -> Tuple[List[EntitySeed], List[Observation]]:
    if not seeds or not cam_obs:
        return seeds, cam_obs
    cost = np.full((len(cam_obs), len(seeds)), float("inf"), dtype=float)
    for i, c in enumerate(cam_obs):
        for j, seed in enumerate(seeds):
            cost[i, j] = camera_attachment_cost(seed, c, cfg)
    matches = _hungarian_assign(cost)
    used_c = set()
    for i, j in matches:
        seeds[j].add(cam_obs[i])
        used_c.add(i)
    unmatched = [obs for i, obs in enumerate(cam_obs) if i not in used_c]
    return seeds, unmatched
