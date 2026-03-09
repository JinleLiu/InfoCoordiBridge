from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


def _to_list(x: Optional[np.ndarray | Sequence[float]]) -> Optional[List[float]]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    return [float(v) for v in x]


@dataclass
class CameraCalibration:
    """Calibration for a single camera view."""

    camera_name: str
    sensor_to_ego: np.ndarray  # 4x4
    intrinsics: np.ndarray     # 3x3
    image_size: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "sensor_to_ego": np.asarray(self.sensor_to_ego, dtype=float).tolist(),
            "intrinsics": np.asarray(self.intrinsics, dtype=float).tolist(),
            "image_size": list(self.image_size),
        }


@dataclass
class Observation:
    """Unified structured fact for one detection.

    coord_frame:
      - 'ego'   : position / velocity already in ego-BEV frame.
      - 'sensor': position / velocity in sensor frame, needs sensor_to_ego.
      - 'image' : camera-only semantics, uses bbox_2d + camera_calib.
    """

    local_id: str
    modality: str
    sensor_name: str
    coord_frame: str
    score: float
    class_probs: Dict[str, float]
    timestamp_us: Optional[int] = None
    track_id: Optional[str] = None
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    size: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    covariance: Optional[np.ndarray] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    bbox_2d: Optional[Tuple[float, float, float, float]] = None
    camera_name: Optional[str] = None
    camera_calib: Optional[CameraCalibration] = None
    sensor_to_ego: Optional[np.ndarray] = None
    image_size: Optional[Tuple[int, int]] = None
    lineage: List[str] = field(default_factory=list)

    @property
    def class_name(self) -> str:
        if not self.class_probs:
            return "unknown"
        return max(self.class_probs.items(), key=lambda kv: kv[1])[0]

    @property
    def class_confidence(self) -> float:
        if not self.class_probs:
            return float(self.score)
        return float(max(self.class_probs.values()))

    @property
    def uncertainty_scalar(self) -> float:
        if self.covariance is not None:
            cov = np.asarray(self.covariance, dtype=float)
            if cov.ndim == 1:
                return float(np.mean(np.abs(cov)))
            return float(np.trace(cov[: min(3, cov.shape[0]), : min(3, cov.shape[1])]))
        return float(max(1e-3, 1.0 - self.score))

    def copy(self) -> "Observation":
        return Observation(
            local_id=self.local_id,
            modality=self.modality,
            sensor_name=self.sensor_name,
            coord_frame=self.coord_frame,
            score=float(self.score),
            class_probs=dict(self.class_probs),
            timestamp_us=self.timestamp_us,
            track_id=self.track_id,
            position=None if self.position is None else np.array(self.position, dtype=float).copy(),
            velocity=None if self.velocity is None else np.array(self.velocity, dtype=float).copy(),
            size=None if self.size is None else np.array(self.size, dtype=float).copy(),
            yaw=None if self.yaw is None else float(self.yaw),
            covariance=None if self.covariance is None else np.array(self.covariance, dtype=float).copy(),
            attrs=dict(self.attrs),
            bbox_2d=None if self.bbox_2d is None else tuple(float(v) for v in self.bbox_2d),
            camera_name=self.camera_name,
            camera_calib=self.camera_calib,
            sensor_to_ego=None if self.sensor_to_ego is None else np.array(self.sensor_to_ego, dtype=float).copy(),
            image_size=self.image_size,
            lineage=list(self.lineage),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_id": self.local_id,
            "track_id": self.track_id,
            "modality": self.modality,
            "sensor_name": self.sensor_name,
            "coord_frame": self.coord_frame,
            "score": float(self.score),
            "class_name": self.class_name,
            "class_confidence": self.class_confidence,
            "class_probs": {str(k): float(v) for k, v in self.class_probs.items()},
            "position": _to_list(self.position),
            "velocity": _to_list(self.velocity),
            "size": _to_list(self.size),
            "yaw": None if self.yaw is None else float(self.yaw),
            "covariance": None if self.covariance is None else np.asarray(self.covariance, dtype=float).tolist(),
            "attrs": self.attrs,
            "bbox_2d": None if self.bbox_2d is None else [float(v) for v in self.bbox_2d],
            "camera_name": self.camera_name,
            "image_size": None if self.image_size is None else list(self.image_size),
            "lineage": list(self.lineage),
        }


@dataclass
class EgoState:
    position_bev_m: Tuple[float, float, float]
    velocity_bev_mps: Tuple[float, float, float]
    yaw_rad: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_bev_m": [float(v) for v in self.position_bev_m],
            "velocity_bev_mps": [float(v) for v in self.velocity_bev_mps],
            "yaw_rad": float(self.yaw_rad),
        }


@dataclass
class FusedEntity:
    entity_id: str
    category: str
    category_confidence: float
    position_bev_m: np.ndarray
    velocity_bev_mps: np.ndarray
    yaw_rad: Optional[float]
    size_m: np.ndarray
    covariance: Optional[np.ndarray]
    ambiguity_flags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    fusion_lineage: List[str] = field(default_factory=list)
    source_observation_ids: List[str] = field(default_factory=list)
    semantic_attributes: Dict[str, Any] = field(default_factory=dict)
    raw_observations: List[Observation] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def uncertainty_scalar(self) -> Optional[float]:
        if self.covariance is None:
            return None
        cov = np.asarray(self.covariance, dtype=float)
        if cov.ndim == 1:
            return float(np.mean(np.abs(cov)))
        return float(np.trace(cov[: min(4, cov.shape[0]), : min(4, cov.shape[1])]))

    def motion_state(self, speed_thresh: float = 0.5) -> str:
        speed = float(np.linalg.norm(self.velocity_bev_mps[:2]))
        return "moving" if speed >= speed_thresh else "stationary"

    def to_dict(self, coordiworld_view: bool = True) -> Dict[str, Any]:
        base = {
            "entity_id": self.entity_id,
            "category": self.category,
            "class_confidence": float(self.category_confidence),
            "position_bev_m": _to_list(self.position_bev_m),
            "velocity_bev_mps": _to_list(self.velocity_bev_mps),
            "yaw_rad": None if self.yaw_rad is None else float(self.yaw_rad),
            "size_m": _to_list(self.size_m),
            "covariance": None if self.covariance is None else np.asarray(self.covariance, dtype=float).tolist(),
            "uncertainty_scalar": self.uncertainty_scalar,
            "ambiguity_flags": list(self.ambiguity_flags),
            "provenance": self.provenance,
            "fusion_lineage": list(self.fusion_lineage),
            "source_observation_ids": list(self.source_observation_ids),
            "semantic_attributes": dict(self.semantic_attributes),
            "motion_state": self.motion_state(),
        }
        if not coordiworld_view:
            base["extras"] = self.extras
            base["raw_observations"] = [obs.to_dict() for obs in self.raw_observations]
        else:
            base["provenance"] = {
                "source_modalities": self.provenance.get("source_modalities", []),
                "conflict_resolved": self.provenance.get("conflict_resolved", {}),
                "single_source_retained": self.provenance.get("single_source_retained", False),
                "camera_semantic_attached": self.provenance.get("camera_semantic_attached", False),
            }
        return base


@dataclass
class SceneSummary:
    scene_token: str
    sample_token: str
    timestamp_us: int
    ego_state: EgoState
    agents: List[FusedEntity]
    map_context: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, coordiworld_view: bool = True) -> Dict[str, Any]:
        return {
            "scene_token": self.scene_token,
            "sample_token": self.sample_token,
            "timestamp_us": int(self.timestamp_us),
            "ego_state": self.ego_state.to_dict(),
            "agents": [a.to_dict(coordiworld_view=coordiworld_view) for a in self.agents],
            "map_context": self.map_context,
            "metadata": dict(self.metadata),
        }

    def to_coordiworld_dict(self) -> Dict[str, Any]:
        return self.to_dict(coordiworld_view=True)
