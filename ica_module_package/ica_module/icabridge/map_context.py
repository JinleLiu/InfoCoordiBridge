from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np

from .geometry import global_to_ego_xy


class NuScenesMapExtractor:
    """Optional nuScenes map extractor.

    This module is intentionally lightweight. If the devkit or map expansion data are not present,
    it falls back to an empty map context so the ICA pipeline still works.
    """

    def __init__(self, dataroot: str):
        self.dataroot = dataroot
        self._maps: Dict[str, Any] = {}
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap  # type: ignore
            self._NuScenesMap = NuScenesMap
        except Exception:
            self._NuScenesMap = None

    def _get_map(self, location: str):
        if self._NuScenesMap is None:
            return None
        if location not in self._maps:
            self._maps[location] = self._NuScenesMap(dataroot=self.dataroot, map_name=location)
        return self._maps[location]

    @staticmethod
    def _yaw_to_global_rot(yaw: float) -> np.ndarray:
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        return np.array([[c, -s], [s, c]], dtype=float)

    def extract(self,
                location: str,
                ego_translation_global: Sequence[float],
                ego_yaw_global: float,
                radius_m: float = 50.0) -> Dict[str, Any]:
        nmap = self._get_map(location)
        if nmap is None:
            return {
                "location": location,
                "closest_lane_token": None,
                "nearby_lanes": [],
                "ped_crossings": [],
                "stop_lines": [],
                "drivable_areas": [],
            }

        x, y = float(ego_translation_global[0]), float(ego_translation_global[1])
        out: Dict[str, Any] = {
            "location": location,
            "closest_lane_token": None,
            "nearby_lanes": [],
            "ped_crossings": [],
            "stop_lines": [],
            "drivable_areas": [],
        }

        try:
            out["closest_lane_token"] = nmap.get_closest_lane(x, y, radius=5.0)
        except Exception:
            out["closest_lane_token"] = None

        # Lanes and lane connectors.
        try:
            lane_records = nmap.get_records_in_radius(x, y, radius_m, ['lane', 'lane_connector'])
            lane_tokens = lane_records.get('lane', []) + lane_records.get('lane_connector', [])
            discrete = nmap.discretize_lanes(lane_tokens, 1.0)
            lanes = []
            for token, pts in discrete.items():
                pts = np.asarray(pts, dtype=float)
                if pts.ndim == 2 and pts.shape[1] >= 2:
                    bev = global_to_ego_xy(pts[:, :2], ego_translation_global, ego_yaw_global)
                    lanes.append({
                        "token": token,
                        "layer": "lane_connector" if token in lane_records.get('lane_connector', []) else "lane",
                        "points_bev": bev.astype(float).tolist(),
                    })
            out["nearby_lanes"] = lanes
        except Exception:
            pass

        def _extract_polygons(layer_name: str, out_key: str):
            try:
                records = nmap.get_records_in_radius(x, y, radius_m, [layer_name])
                tokens = records.get(layer_name, [])
                polys = []
                for token in tokens:
                    rec = nmap.get(layer_name, token)
                    poly_token = rec.get('polygon_token')
                    if not poly_token:
                        continue
                    poly = nmap.extract_polygon(poly_token)
                    coords = np.asarray(poly.exterior.coords, dtype=float)
                    bev = global_to_ego_xy(coords[:, :2], ego_translation_global, ego_yaw_global)
                    polys.append({
                        "token": token,
                        "layer": layer_name,
                        "polygon_bev": bev.astype(float).tolist(),
                    })
                out[out_key] = polys
            except Exception:
                pass

        _extract_polygons('ped_crossing', 'ped_crossings')
        _extract_polygons('stop_line', 'stop_lines')
        _extract_polygons('drivable_area', 'drivable_areas')
        return out
