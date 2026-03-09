from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import numpy as np

from .association import EntitySeed, stage1_build_seeds, stage2_attach_radar, stage3_attach_camera
from .config import ICAConfig
from .fusion import ReliabilityBank, fuse_seed
from .geometry import point_sensor_to_ego, vector_sensor_to_ego, relative_direction_label
from .map_context import NuScenesMapExtractor
from .schemas import EgoState, Observation, SceneSummary


@dataclass
class ICAResult:
    scene_summary: SceneSummary
    unmatched_geometry: List[Observation]
    unmatched_camera: List[Observation]


class ICAPipeline:
    """Deterministic implementation of the ICA module.

    Expected inputs are already standardized structured facts from the upstream agents.
    That matches the manuscript: ICA operates on structured facts F_k and not on free-form synopses.
    """

    def __init__(self,
                 config: Optional[ICAConfig] = None,
                 reliability_bank: Optional[ReliabilityBank] = None,
                 map_extractor: Optional[NuScenesMapExtractor] = None):
        self.cfg = config or ICAConfig()
        self.reliability_bank = reliability_bank or ReliabilityBank()
        self.map_extractor = map_extractor

    def _normalize_observation(self, obs: Observation) -> Observation:
        obs = obs.copy()
        modality = obs.modality.lower()
        if obs.coord_frame == "sensor":
            if obs.sensor_to_ego is None:
                raise ValueError(f"Observation {obs.local_id} is in sensor frame but sensor_to_ego is missing.")
            if obs.position is not None:
                obs.position = point_sensor_to_ego(obs.position, obs.sensor_to_ego)
            if obs.velocity is not None:
                obs.velocity = vector_sensor_to_ego(obs.velocity, obs.sensor_to_ego)
            obs.coord_frame = "ego"
        elif obs.coord_frame == "ego":
            pass
        elif obs.coord_frame == "image":
            pass
        else:
            raise ValueError(f"Unknown coord_frame: {obs.coord_frame}")

        if obs.lineage is None or len(obs.lineage) == 0:
            obs.lineage = [obs.modality]
        return obs

    def _partition_observations(self, observations: Iterable[Observation]) -> Tuple[List[Observation], List[Observation], List[Observation], List[Observation]]:
        lidar_obs, bev_obs, radar_obs, cam_obs = [], [], [], []
        for obs in observations:
            norm = self._normalize_observation(obs)
            m = norm.modality.lower()
            if m == "lidar":
                lidar_obs.append(norm)
            elif m == "bevfusion":
                bev_obs.append(norm)
            elif m == "radar":
                radar_obs.append(norm)
            elif m == "camera":
                cam_obs.append(norm)
            else:
                raise ValueError(f"Unsupported modality: {obs.modality}")
        return lidar_obs, bev_obs, radar_obs, cam_obs

    def _single_source_seeds(self, unmatched: List[Observation], start_idx: int) -> List[EntitySeed]:
        seeds = []
        for i, obs in enumerate(unmatched):
            seeds.append(EntitySeed(seed_id=f"seed_{start_idx + i:04d}", observations=[obs]))
        return seeds

    def compile(self,
                observations: Iterable[Observation],
                ego_state: EgoState,
                scene_token: str,
                sample_token: str,
                timestamp_us: int,
                metadata: Optional[Dict[str, Any]] = None,
                location: Optional[str] = None,
                ego_translation_global: Optional[Sequence[float]] = None,
                ego_yaw_global: Optional[float] = None) -> ICAResult:
        lidar_obs, bev_obs, radar_obs, cam_obs = self._partition_observations(observations)

        # Stage 1: LiDAR <-> BEVFusion geometric seeds.
        seeds = stage1_build_seeds(lidar_obs, bev_obs, self.cfg)

        # Stage 2: Radar -> seeds.
        seeds, unmatched_radar = stage2_attach_radar(seeds, radar_obs, self.cfg)

        # Keep unmatched radar as single-source entities.
        next_idx = len(seeds)
        seeds.extend(self._single_source_seeds(unmatched_radar, next_idx))
        next_idx = len(seeds)

        # Stage 3: Camera semantics -> entities.
        seeds, unmatched_camera = stage3_attach_camera(seeds, cam_obs, self.cfg)

        # Fuse each seed.
        fused_entities = [fuse_seed(seed, self.cfg, self.reliability_bank) for seed in seeds]
        fused_entities = fused_entities[: self.cfg.max_entities]

        # Derived relation- and zone-style fields that are useful for Paper 2 and harmless for Paper 3.
        closest_lane = None
        map_context = {
            "location": location,
            "closest_lane_token": None,
            "nearby_lanes": [],
            "ped_crossings": [],
            "stop_lines": [],
            "drivable_areas": [],
        }
        if self.map_extractor is not None and location is not None and ego_translation_global is not None and ego_yaw_global is not None:
            map_context = self.map_extractor.extract(location, ego_translation_global, ego_yaw_global, radius_m=self.cfg.map_radius_m)
            closest_lane = map_context.get("closest_lane_token")

        for ent in fused_entities:
            pos = np.asarray(ent.position_bev_m, dtype=float)
            ent.semantic_attributes.setdefault("relative_position_to_ego", relative_direction_label(pos[:2]))
            ent.semantic_attributes.setdefault("motion_state", ent.motion_state(self.cfg.motion_speed_threshold_mps))
            # zone heuristic
            zone = "normal_lane"
            for poly in map_context.get("ped_crossings", []):
                pts = np.asarray(poly.get("polygon_bev", []), dtype=float)
                if len(pts) == 0:
                    continue
                d = np.linalg.norm(pts[:, :2] - pos[:2], axis=1).min()
                if d < 3.0:
                    zone = "crosswalk_vicinity"
                    break
            if zone == "normal_lane" and closest_lane and abs(float(pos[1])) < 4.0 and float(pos[0]) > 0.0:
                zone = "ego_lane_front"
            ent.semantic_attributes.setdefault("zone", zone)
            if closest_lane and abs(float(pos[1])) < 2.5:
                ent.semantic_attributes.setdefault("lane_relation", "ego_lane")

        scene_summary = SceneSummary(
            scene_token=scene_token,
            sample_token=sample_token,
            timestamp_us=int(timestamp_us),
            ego_state=ego_state,
            agents=fused_entities,
            map_context=map_context,
            metadata=metadata or {},
        )

        unmatched_geometry = []
        return ICAResult(scene_summary=scene_summary,
                         unmatched_geometry=unmatched_geometry,
                         unmatched_camera=unmatched_camera)

    @staticmethod
    def save_json(scene_summary: SceneSummary, out_path: str, coordiworld_view: bool = True) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scene_summary.to_dict(coordiworld_view=coordiworld_view), f, ensure_ascii=False, indent=2)
