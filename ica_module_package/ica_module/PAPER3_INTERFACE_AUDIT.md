# Paper 3 interface audit: What CoordiWorld actually needs from InfoCoordiBridge / ICA

The earlier Paper 3 design only requires a **subset** of the full InfoCoordiBridge output.
This package therefore exports `SceneSummary.to_coordiworld_dict()` as the default JSON view.

## Required by Paper 3

### 1. Frame-level metadata
- `scene_token`
- `sample_token`
- `timestamp_us`

### 2. Ego state
- `ego_state.position_bev_m`
- `ego_state.velocity_bev_mps`
- `ego_state.yaw_rad`

### 3. Dynamic agents (entity tokens)
For each entity:
- `entity_id`
- `category`
- `class_confidence`
- `position_bev_m`
- `velocity_bev_mps`
- `yaw_rad`
- `size_m`
- `covariance`
- `uncertainty_scalar`
- `ambiguity_flags`
- `provenance.source_modalities`
- `provenance.conflict_resolved`
- `provenance.single_source_retained`
- `provenance.camera_semantic_attached`
- `fusion_lineage`

### 4. Static map context
- `map_context.closest_lane_token`
- `map_context.nearby_lanes`
- `map_context.ped_crossings`
- `map_context.stop_lines`
- `map_context.drivable_areas`

## Optional extras retained for Paper 2 / debugging
- `semantic_attributes.fine_type`
- `semantic_attributes.color`
- `semantic_attributes.intent`
- `semantic_attributes.motion_state`
- `semantic_attributes.relative_position_to_ego`
- `semantic_attributes.zone`

## Export choice in this codebase
- `scene_summary.to_coordiworld_dict()`  -> Paper 3 oriented subset (default)
- `scene_summary.to_dict(coordiworld_view=False)` -> fuller ICA / Paper 2 oriented export
