from __future__ import annotations

import numpy as np

from icabridge import ICAConfig, ICAPipeline, Observation, EgoState


def test_pipeline_smoke():
    ego = EgoState(position_bev_m=(0.0, 0.0, 0.0), velocity_bev_mps=(0.0, 0.0, 0.0), yaw_rad=0.0)
    obs = [
        Observation(
            local_id="l1",
            modality="lidar",
            sensor_name="LIDAR_TOP",
            coord_frame="ego",
            score=0.95,
            class_probs={"car": 1.0},
            track_id="track_1",
            position=np.array([10.0, 0.5, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            size=np.array([4.5, 1.9, 1.6]),
            yaw=0.0,
            covariance=np.diag([0.1**2, 0.1**2, 0.2**2, 0.2**2]),
        ),
        Observation(
            local_id="b1",
            modality="bevfusion",
            sensor_name="BEVFusion",
            coord_frame="ego",
            score=0.90,
            class_probs={"van": 1.0},
            track_id="track_1",
            position=np.array([10.2, 0.6, 0.0]),
            velocity=np.array([1.1, -0.1, 0.0]),
            size=np.array([4.8, 2.0, 1.8]),
            yaw=0.03,
            covariance=np.diag([0.2**2, 0.2**2, 0.3**2, 0.3**2]),
            lineage=["camera", "lidar", "bevfusion"],
        ),
    ]
    pipeline = ICAPipeline(ICAConfig())
    result = pipeline.compile(obs, ego, "scene", "sample", 0)
    out = result.scene_summary.to_coordiworld_dict()
    assert len(out["agents"]) >= 1
    assert out["agents"][0]["entity_id"] == "track_1"
