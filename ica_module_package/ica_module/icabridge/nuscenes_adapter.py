from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random
import numpy as np

from .geometry import global_to_ego_xy, make_transform, normalize_angle, rotation_matrix_z, transform_points, yaw_from_rotation_matrix
from .map_context import NuScenesMapExtractor
from .pipeline import ICAPipeline
from .schemas import CameraCalibration, EgoState, Observation, SceneSummary


NUSC_CANONICAL_MAP = {
    "vehicle.car": "car",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}


COARSE_TO_FINE_CAMERA_VARIANTS = {
    "car": ["car", "sedan", "suv", "van", "mpv"],
    "bus": ["bus"],
    "truck": ["truck"],
    "pedestrian": ["pedestrian"],
    "bicycle": ["bicycle"],
    "motorcycle": ["motorcycle"],
    "trailer": ["trailer"],
    "construction_vehicle": ["construction_vehicle"],
}


class NuScenesICAExampleBuilder:
    """Builds a runnable ICA demo from nuScenes data.

    The purpose is practical integration and debugging, not exact replication of the detectors used in
    the manuscript. Upstream detector outputs are approximated from nuScenes annotations/boxes so that
    the deterministic ICA module can be exercised end-to-end.
    """

    def __init__(self, nusc: Any, dataroot: str, seed: int = 42):
        self.nusc = nusc
        self.dataroot = dataroot
        self.rng = random.Random(seed)
        self.map_extractor = NuScenesMapExtractor(dataroot)

    @staticmethod
    def canonical_category(raw: str) -> str:
        if raw in NUSC_CANONICAL_MAP:
            return NUSC_CANONICAL_MAP[raw]
        # coarse fallback
        for k, v in NUSC_CANONICAL_MAP.items():
            if raw.startswith(k):
                return v
        if raw.startswith("human.pedestrian"):
            return "pedestrian"
        if raw.startswith("vehicle."):
            return raw.split(".")[-1]
        return raw.split(".")[-1]

    @staticmethod
    def _quat_to_yaw(q) -> float:
        # q may be pyquaternion.Quaternion or [w, x, y, z] depending on the devkit record.
        try:
            rot = q.rotation_matrix  # pyquaternion
            return yaw_from_rotation_matrix(rot)
        except Exception:
            from pyquaternion import Quaternion  # type: ignore
            qq = Quaternion(q)
            return yaw_from_rotation_matrix(qq.rotation_matrix)

    @staticmethod
    def _transform_global_to_ego_xyyaw(global_xyz: Sequence[float], global_yaw: float,
                                       ego_translation: Sequence[float], ego_yaw: float) -> Tuple[np.ndarray, float]:
        pos_bev = global_to_ego_xy(np.asarray(global_xyz, dtype=float), ego_translation, ego_yaw)[0]
        z = float(global_xyz[2] - ego_translation[2]) if len(global_xyz) >= 3 else 0.0
        yaw_ego = normalize_angle(global_yaw - ego_yaw)
        return np.array([pos_bev[0], pos_bev[1], z], dtype=float), yaw_ego

    def _ego_pose_for_sample_data(self, sample_data_token: str) -> Tuple[np.ndarray, float, Dict[str, Any], Dict[str, Any]]:
        sd = self.nusc.get('sample_data', sample_data_token)
        ego_pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
        from pyquaternion import Quaternion  # type: ignore
        yaw = yaw_from_rotation_matrix(Quaternion(ego_pose['rotation']).rotation_matrix)
        return np.asarray(ego_pose['translation'], dtype=float), float(yaw), sd, ego_pose

    def _calib_sensor_to_ego(self, sample_data_token: str) -> np.ndarray:
        sd = self.nusc.get('sample_data', sample_data_token)
        calib = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        from pyquaternion import Quaternion  # type: ignore
        R = Quaternion(calib['rotation']).rotation_matrix
        t = np.asarray(calib['translation'], dtype=float)
        return make_transform(R, t)

    def _camera_calibration(self, cam_token: str) -> CameraCalibration:
        sd = self.nusc.get('sample_data', cam_token)
        calib = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        from pyquaternion import Quaternion  # type: ignore
        R = Quaternion(calib['rotation']).rotation_matrix
        T_sensor_to_ego = make_transform(R, np.asarray(calib['translation'], dtype=float))
        K = np.asarray(calib['camera_intrinsic'], dtype=float)
        image_size = (int(sd.get('width', 1600)), int(sd.get('height', 900)))
        return CameraCalibration(camera_name=sd['channel'], sensor_to_ego=T_sensor_to_ego, intrinsics=K, image_size=image_size)

    def _ego_velocity(self, sample_token: str, reference_channel: str = 'LIDAR_TOP') -> np.ndarray:
        sample = self.nusc.get('sample', sample_token)
        sd_tok = sample['data'][reference_channel]
        curr_xyz, curr_yaw, _, curr_pose = self._ego_pose_for_sample_data(sd_tok)
        prev_token = sample.get('prev')
        next_token = sample.get('next')
        if prev_token:
            prev_sample = self.nusc.get('sample', prev_token)
            prev_xyz, prev_yaw, _, prev_pose = self._ego_pose_for_sample_data(prev_sample['data'][reference_channel])
            dt = max(1e-3, (curr_pose['timestamp'] - prev_pose['timestamp']) / 1e6)
            v_global = (curr_xyz - prev_xyz) / dt
        elif next_token:
            next_sample = self.nusc.get('sample', next_token)
            next_xyz, next_yaw, _, next_pose = self._ego_pose_for_sample_data(next_sample['data'][reference_channel])
            dt = max(1e-3, (next_pose['timestamp'] - curr_pose['timestamp']) / 1e6)
            v_global = (next_xyz - curr_xyz) / dt
        else:
            v_global = np.zeros(3, dtype=float)
        c, s = math.cos(-curr_yaw), math.sin(-curr_yaw)
        R = np.array([[c, -s], [s, c]], dtype=float)
        v_xy = (R @ v_global[:2].reshape(2, 1)).reshape(2)
        return np.array([v_xy[0], v_xy[1], v_global[2]], dtype=float)

    def _annotation_to_ego(self, ann_token: str, ego_translation: np.ndarray, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, str, Dict[str, Any]]:
        ann = self.nusc.get('sample_annotation', ann_token)
        global_xyz = np.asarray(ann['translation'], dtype=float)
        # nuScenes stores size as [w, l, h]; convert to [l, w, h] for our representation.
        size = np.array([ann['size'][1], ann['size'][0], ann['size'][2]], dtype=float)
        from pyquaternion import Quaternion  # type: ignore
        yaw_global = yaw_from_rotation_matrix(Quaternion(ann['rotation']).rotation_matrix)
        pos_ego, yaw_ego = self._transform_global_to_ego_xyyaw(global_xyz, yaw_global, ego_translation, ego_yaw)
        v_global = np.asarray(self.nusc.box_velocity(ann_token), dtype=float)
        if np.any(np.isnan(v_global)):
            v_global = np.zeros(3, dtype=float)
        c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
        R = np.array([[c, -s], [s, c]], dtype=float)
        v_xy = (R @ v_global[:2].reshape(2, 1)).reshape(2)
        v_ego = np.array([v_xy[0], v_xy[1], v_global[2]], dtype=float)
        category = self.canonical_category(ann['category_name'])
        extras = {
            "instance_token": ann['instance_token'],
            "visibility_token": ann.get('visibility_token'),
            "raw_category_name": ann['category_name'],
            "num_lidar_pts": ann.get('num_lidar_pts'),
            "num_radar_pts": ann.get('num_radar_pts'),
        }
        return pos_ego, size, yaw_ego, v_ego, category, extras

    def _maybe_bevfusion_variant(self, coarse_category: str) -> str:
        if coarse_category == 'car' and self.rng.random() < 0.25:
            return self.rng.choice(['van', 'mpv', 'suv'])
        return coarse_category

    def _camera_bbox_from_global_ann(self, ann_token: str, cam_token: str) -> Optional[Tuple[Tuple[float, float, float, float], Dict[str, Any]]]:
        try:
            from nuscenes.utils.geometry_utils import view_points  # type: ignore
            from nuscenes.utils.data_classes import Box  # type: ignore
            from pyquaternion import Quaternion  # type: ignore
        except Exception:
            return None
        ann = self.nusc.get('sample_annotation', ann_token)
        sd = self.nusc.get('sample_data', cam_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', sd['ego_pose_token'])
        box = self.nusc.get_box(ann_token)
        box.translate(-np.asarray(ego_pose['translation'], dtype=float))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        box.translate(-np.asarray(cs['translation'], dtype=float))
        box.rotate(Quaternion(cs['rotation']).inverse)
        corners = box.corners()
        valid = corners[2, :] > 1e-3
        if valid.sum() == 0:
            return None
        intrinsic = np.asarray(cs['camera_intrinsic'], dtype=float)
        pts = view_points(corners[:, valid], intrinsic, normalize=True)
        x1, y1 = float(np.min(pts[0])), float(np.min(pts[1]))
        x2, y2 = float(np.max(pts[0])), float(np.max(pts[1]))
        w, h = int(sd.get('width', 1600)), int(sd.get('height', 900))
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return (float(x1), float(y1), float(x2), float(y2)), {
            "raw_category_name": ann['category_name'],
            "instance_token": ann['instance_token'],
        }

    def build_inputs(self,
                     sample_token: str,
                     camera_channels: Optional[Sequence[str]] = None,
                     simulate_conflicts: bool = True) -> Tuple[List[Observation], EgoState, Dict[str, Any]]:
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']

        lidar_token = sample['data']['LIDAR_TOP']
        ego_translation, ego_yaw, _, _ = self._ego_pose_for_sample_data(lidar_token)
        ego_velocity = self._ego_velocity(sample_token, 'LIDAR_TOP')
        ego_state = EgoState(position_bev_m=(0.0, 0.0, 0.0),
                             velocity_bev_mps=(float(ego_velocity[0]), float(ego_velocity[1]), float(ego_velocity[2])),
                             yaw_rad=0.0)
        observations: List[Observation] = []

        # LiDAR + BEVFusion + Radar pseudo outputs from annotations.
        for ann_token in sample['anns']:
            pos_ego, size, yaw_ego, vel_ego, category, extras = self._annotation_to_ego(ann_token, ego_translation, ego_yaw)
            track_id = extras['instance_token']
            class_probs = {category: 1.0}
            obs_lidar = Observation(
                local_id=f"lidar_{ann_token[:8]}",
                modality="lidar",
                sensor_name="LIDAR_TOP",
                coord_frame="ego",
                score=0.95,
                class_probs=class_probs,
                track_id=track_id,
                position=pos_ego.copy(),
                velocity=vel_ego.copy(),
                size=size.copy(),
                yaw=float(yaw_ego),
                covariance=np.diag([0.15**2, 0.15**2, 0.25**2, 0.25**2]),
                attrs={},
                lineage=["lidar"],
            )
            observations.append(obs_lidar)

            # BEVFusion pseudo detector: mild noise + possible class conflict.
            bev_category = self._maybe_bevfusion_variant(category) if simulate_conflicts else category
            pos_bev = pos_ego.copy()
            vel_bev = vel_ego.copy()
            if simulate_conflicts:
                pos_bev[:2] += np.array([self.rng.uniform(-0.35, 0.35), self.rng.uniform(-0.35, 0.35)])
                vel_bev[:2] += np.array([self.rng.uniform(-0.15, 0.15), self.rng.uniform(-0.15, 0.15)])
            obs_bev = Observation(
                local_id=f"bev_{ann_token[:8]}",
                modality="bevfusion",
                sensor_name="BEVFusion",
                coord_frame="ego",
                score=0.90,
                class_probs={bev_category: 1.0},
                track_id=track_id,
                position=pos_bev,
                velocity=vel_bev,
                size=size.copy(),
                yaw=float(yaw_ego + (self.rng.uniform(-0.05, 0.05) if simulate_conflicts else 0.0)),
                covariance=np.diag([0.25**2, 0.25**2, 0.35**2, 0.35**2]),
                attrs={},
                lineage=["camera", "lidar", "bevfusion"],
            )
            observations.append(obs_bev)

            # Radar pseudo detections only for dynamic / near objects.
            speed = float(np.linalg.norm(vel_ego[:2]))
            dist = float(np.linalg.norm(pos_ego[:2]))
            if dist <= 80.0 and speed >= 0.15:
                radar_pos = pos_ego.copy()
                radar_vel = vel_ego.copy()
                if simulate_conflicts:
                    radar_pos[:2] += np.array([self.rng.uniform(-0.60, 0.60), self.rng.uniform(-0.60, 0.60)])
                    radar_vel[:2] += np.array([self.rng.uniform(-0.20, 0.20), self.rng.uniform(-0.20, 0.20)])
                obs_radar = Observation(
                    local_id=f"radar_{ann_token[:8]}",
                    modality="radar",
                    sensor_name="RADAR_COMBINED",
                    coord_frame="ego",
                    score=0.85,
                    class_probs={category: 1.0},
                    track_id=track_id,
                    position=radar_pos,
                    velocity=radar_vel,
                    size=size.copy(),
                    yaw=None,
                    covariance=np.diag([0.45**2, 0.45**2, 0.20**2, 0.20**2]),
                    attrs={},
                    lineage=["radar"],
                )
                observations.append(obs_radar)

        # Camera semantic-only facts.
        if camera_channels is None:
            camera_channels = [
                'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]
        for ch in camera_channels:
            if ch not in sample['data']:
                continue
            cam_token = sample['data'][ch]
            calib = self._camera_calibration(cam_token)
            for ann_token in sample['anns']:
                out = self._camera_bbox_from_global_ann(ann_token, cam_token)
                if out is None:
                    continue
                bbox, extras = out
                ann = self.nusc.get('sample_annotation', ann_token)
                coarse = self.canonical_category(ann['category_name'])
                fine_type = self._maybe_bevfusion_variant(coarse) if coarse == 'car' and simulate_conflicts else coarse
                attrs = {"fine_type": fine_type}
                obs_cam = Observation(
                    local_id=f"cam_{ch}_{ann_token[:8]}",
                    modality="camera",
                    sensor_name=ch,
                    coord_frame="image",
                    score=0.88,
                    class_probs={coarse: 1.0},
                    track_id=ann['instance_token'],
                    bbox_2d=bbox,
                    camera_name=ch,
                    camera_calib=calib,
                    image_size=calib.image_size,
                    attrs=attrs,
                    lineage=["camera"],
                )
                observations.append(obs_cam)

        meta = {
            "scene_token": scene['token'],
            "scene_name": scene['name'],
            "location": location,
            "sample_token": sample_token,
            "timestamp_us": int(sample['timestamp']),
            "ego_translation_global": ego_translation.astype(float).tolist(),
            "ego_yaw_global": float(ego_yaw),
        }
        return observations, ego_state, meta

    def run_pipeline(self,
                     pipeline: ICAPipeline,
                     sample_token: str,
                     camera_channels: Optional[Sequence[str]] = None,
                     simulate_conflicts: bool = True):
        obs, ego_state, meta = self.build_inputs(sample_token, camera_channels=camera_channels, simulate_conflicts=simulate_conflicts)
        result = pipeline.compile(
            observations=obs,
            ego_state=ego_state,
            scene_token=meta['scene_token'],
            sample_token=meta['sample_token'],
            timestamp_us=meta['timestamp_us'],
            metadata={"scene_name": meta['scene_name'], "location": meta['location'], "source": "nuScenes-demo"},
            location=meta['location'],
            ego_translation_global=meta['ego_translation_global'],
            ego_yaw_global=meta['ego_yaw_global'],
        )
        return result
