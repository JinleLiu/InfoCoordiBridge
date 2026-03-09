from __future__ import annotations

import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from icabridge import ICAConfig, ICAPipeline, Observation, EgoState


def build_case_a_inputs():
    ego = EgoState(position_bev_m=(0.0, 0.0, 0.0), velocity_bev_mps=(0.0, 0.0, 0.0), yaw_rad=0.0)
    obs = []

    # Lead vehicle: LiDAR says SUV-like, BEVFusion says van, camera says MPV.
    obs.append(Observation(
        local_id="lidar_lead",
        modality="lidar",
        sensor_name="LIDAR_TOP",
        coord_frame="ego",
        score=0.97,
        class_probs={"car": 0.55, "suv": 0.45},
        track_id="lead_vehicle_track",
        position=np.array([12.2, 0.6, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        size=np.array([4.8, 2.0, 1.8]),
        yaw=0.02,
        covariance=np.diag([0.1**2, 0.1**2, 0.2**2, 0.2**2]),
    ))
    obs.append(Observation(
        local_id="bev_lead",
        modality="bevfusion",
        sensor_name="BEVFusion",
        coord_frame="ego",
        score=0.93,
        class_probs={"van": 0.90, "car": 0.10},
        track_id="lead_vehicle_track",
        position=np.array([12.5, 0.8, 0.0]),
        velocity=np.array([0.9, -0.1, 0.0]),
        size=np.array([5.1, 2.1, 2.0]),
        yaw=0.04,
        covariance=np.diag([0.25**2, 0.25**2, 0.3**2, 0.3**2]),
        lineage=["camera", "lidar", "bevfusion"],
    ))
    obs.append(Observation(
        local_id="cam_lead",
        modality="camera",
        sensor_name="CAM_FRONT",
        coord_frame="image",
        score=0.92,
        class_probs={"car": 1.0},
        track_id="lead_vehicle_track",
        bbox_2d=(500.0, 300.0, 820.0, 620.0),
        attrs={"fine_type": "mpv"},
        # stage3 needs calib, so for synthetic example we skip camera attach and keep the semantic cue in extras later
    ))

    # Pedestrian only seen by camera/lidar-like sparse evidence retained for completeness.
    obs.append(Observation(
        local_id="ped_single",
        modality="lidar",
        sensor_name="LIDAR_TOP",
        coord_frame="ego",
        score=0.81,
        class_probs={"pedestrian": 1.0},
        track_id="ped_track_03",
        position=np.array([18.5, 4.0, 0.0]),
        velocity=np.array([0.2, 0.0, 0.0]),
        size=np.array([0.7, 0.7, 1.7]),
        yaw=0.0,
        covariance=np.diag([0.2**2, 0.2**2, 0.25**2, 0.25**2]),
    ))
    return obs, ego


if __name__ == "__main__":
    cfg = ICAConfig()
    pipeline = ICAPipeline(cfg)
    obs, ego = build_case_a_inputs()
    # Remove camera semantic-only item because synthetic example lacks camera calibration.
    obs = [o for o in obs if o.modality != "camera"]
    result = pipeline.compile(
        observations=obs,
        ego_state=ego,
        scene_token="synthetic_scene",
        sample_token="synthetic_sample",
        timestamp_us=0,
        metadata={"source": "synthetic_demo"},
        location="synthetic_city",
        ego_translation_global=[0.0, 0.0, 0.0],
        ego_yaw_global=0.0,
    )
    print(json.dumps(result.scene_summary.to_coordiworld_dict(), ensure_ascii=False, indent=2))
