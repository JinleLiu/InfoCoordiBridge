# InfoCoordiBridge ICA Module (Deterministic SceneSummary Compiler)

This package implements the **Information Coordination and Abstraction (ICA)** module from
`manuscript-InfoCoordiBridge.pdf` as a standalone, reusable Python component.

It consumes standardized **structured facts** from four upstream perception agents:
- `BEVFusion`
- `LiDAR`
- `Radar`
- `Camera`

and deterministically compiles them into a single, conflict-resolved, provenance-aware
**SceneSummary**.

## What this implementation covers

This code implements the deterministic core described in the paper:

1. **Cross-sensor coordinate normalization and hierarchical entity alignment**
   - LiDAR ↔ BEVFusion seed matching (geometry first)
   - Radar → seeds (motion refinement)
   - Camera semantics → entities via **projection-based attachment** (no camera-depth overwrite)

2. **Conflict-aware attribute fusion**
   - weighted evidence fusion for continuous attributes
   - covariance intersection (CI) for correlated sources
   - weighted voting for categorical attributes
   - ambiguity flags and provenance traces

3. **Deterministic SceneSummary generation**
   - global entity IDs / track IDs (when available)
   - fused BEV position / velocity / yaw / size
   - class label + confidence
   - uncertainty / covariance
   - provenance and fusion lineage
   - camera semantic attributes (when attached)
   - map context (optional, via nuScenes map expansion API)

## What this implementation intentionally does **not** cover

The manuscript separates **multi-agent perception** from **ICA**. Therefore this package starts from
structured JSON-like facts and does **not** train or ship the upstream models themselves
(e.g. InsightGPT, CenterPoint, radar clustering, or BEVFusion).

For a runnable example on nuScenes, the included demo builds **pseudo agent outputs** from nuScenes
annotations / official devkit utilities, then feeds them into ICA.

---

## Install

### Minimal install

```bash
pip install -r requirements.txt
```

### nuScenes example install

You also need the official nuScenes devkit and data.

```bash
pip install nuscenes-devkit
```

Download the dataset and map expansion pack, then place the map folders under your nuScenes
`maps/` directory as described in the official devkit/tutorials.

---

## Quick smoke test (no nuScenes required)

```bash
python examples/synthetic_demo.py
```

This prints a small SceneSummary JSON.

---

## Run on nuScenes

```bash
python examples/nuscenes_ica_demo.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample-token <YOUR_SAMPLE_TOKEN> \
  --out scene_summary_coordiworld.json
```

### Optional flags

- `--full` : export the **full ICA SceneSummary** instead of the CoordiWorld-oriented subset.
- `--no-simulate-conflicts` : disable the mild pseudo-detector disagreement injected for debugging.

---

## Output format

By default the exporter emits the **CoordiWorld-oriented subset** needed by Paper 3:

```json
{
  "scene_token": "...",
  "sample_token": "...",
  "timestamp_us": 0,
  "ego_state": {
    "position_bev_m": [0.0, 0.0, 0.0],
    "velocity_bev_mps": [vx, vy, vz],
    "yaw_rad": 0.0
  },
  "agents": [
    {
      "entity_id": "...",
      "category": "car",
      "class_confidence": 0.91,
      "position_bev_m": [x, y, z],
      "velocity_bev_mps": [vx, vy, vz],
      "yaw_rad": 0.03,
      "size_m": [l, w, h],
      "covariance": [[...]],
      "uncertainty_scalar": 0.12,
      "ambiguity_flags": ["ambiguous_class"],
      "provenance": {
        "source_modalities": ["lidar", "bevfusion", "camera"],
        "conflict_resolved": {"class": ["car", "van"]},
        "single_source_retained": false,
        "camera_semantic_attached": true
      },
      "fusion_lineage": [...],
      "source_observation_ids": [...],
      "semantic_attributes": {
        "fine_type": "mpv",
        "motion_state": "moving",
        "relative_position_to_ego": "front",
        "zone": "ego_lane_front"
      }
    }
  ],
  "map_context": {
    "closest_lane_token": "...",
    "nearby_lanes": [...],
    "ped_crossings": [...],
    "stop_lines": [...],
    "drivable_areas": [...]
  },
  "metadata": {...}
}
```

This is exactly the subset we later use as the upstream interface for **Paper 3 / CoordiWorld**.

---

## Python usage

```python
from icabridge import ICAConfig, ICAPipeline, Observation, EgoState
import numpy as np

cfg = ICAConfig()
pipeline = ICAPipeline(cfg)

ego = EgoState(position_bev_m=(0.0, 0.0, 0.0),
               velocity_bev_mps=(0.0, 0.0, 0.0),
               yaw_rad=0.0)

observations = [
    Observation(
        local_id="lidar_001",
        modality="lidar",
        sensor_name="LIDAR_TOP",
        coord_frame="ego",
        score=0.95,
        class_probs={"car": 1.0},
        track_id="track_001",
        position=np.array([12.0, 0.5, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        size=np.array([4.5, 1.9, 1.6]),
        yaw=0.02,
    )
]

result = pipeline.compile(
    observations=observations,
    ego_state=ego,
    scene_token="scene_x",
    sample_token="sample_x",
    timestamp_us=0,
)

scene_summary = result.scene_summary.to_coordiworld_dict()
```

---

## File structure

```text
ica_module/
├── README.md
├── requirements.txt
├── icabridge/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── geometry.py
│   ├── association.py
│   ├── fusion.py
│   ├── map_context.py
│   ├── pipeline.py
│   └── nuscenes_adapter.py
├── examples/
│   ├── synthetic_demo.py
│   └── nuscenes_ica_demo.py
└── tests/
    └── test_smoke.py
```

---

