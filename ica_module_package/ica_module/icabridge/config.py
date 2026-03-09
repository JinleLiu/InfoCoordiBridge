from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ICAConfig:
    gating_radius_m: float = 4.5
    matching_distance_gate_m: float = 5.0
    class_mismatch_penalty: float = 1.5
    size_penalty_weight: float = 0.25
    velocity_penalty_weight: float = 0.35
    camera_iou_min: float = 0.05
    camera_cost_class_penalty: float = 0.75
    use_covariance_intersection: bool = True
    reliability_beta: float = 0.90
    class_confidence_threshold: float = 0.50
    chi2_gate_threshold: float = 9.21
    ambiguity_residual_threshold_m: float = 2.5
    motion_speed_threshold_mps: float = 0.50
    map_radius_m: float = 50.0
    max_entities: int = 128

    modality_attribute_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "position": {"lidar": 1.00, "bevfusion": 0.90, "radar": 0.60, "camera": 0.05},
        "velocity": {"lidar": 0.70, "bevfusion": 0.65, "radar": 1.00, "camera": 0.10},
        "yaw": {"lidar": 0.95, "bevfusion": 0.85, "radar": 0.25, "camera": 0.10},
        "size": {"lidar": 0.90, "bevfusion": 0.80, "radar": 0.10, "camera": 0.15},
        "class": {"lidar": 0.65, "bevfusion": 0.90, "radar": 0.35, "camera": 0.95},
        "fine_type": {"camera": 1.00, "bevfusion": 0.55, "lidar": 0.40, "radar": 0.10},
        "intent": {"camera": 1.00, "bevfusion": 0.35, "lidar": 0.15, "radar": 0.10},
        "color": {"camera": 1.00, "bevfusion": 0.15, "lidar": 0.05, "radar": 0.05},
    })

    correlated_modalities: Dict[str, set] = field(default_factory=lambda: {
        "bevfusion": {"lidar", "camera"},
        "lidar": {"bevfusion"},
        "camera": {"bevfusion"},
        "radar": set(),
    })

    def prior(self, attribute: str, modality: str) -> float:
        modality = modality.lower()
        return float(self.modality_attribute_priors.get(attribute, {}).get(modality, 0.2))
