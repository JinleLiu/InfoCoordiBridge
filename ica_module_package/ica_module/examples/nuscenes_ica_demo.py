from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from icabridge import ICAConfig, ICAPipeline, NuScenesICAExampleBuilder, NuScenesMapExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Run the deterministic ICA module on a nuScenes sample.")
    parser.add_argument("--dataroot", type=str, required=True, help="Path to nuScenes dataroot, e.g. /data/sets/nuscenes")
    parser.add_argument("--version", type=str, default="v1.0-mini", help="nuScenes version, e.g. v1.0-mini or v1.0-trainval")
    parser.add_argument("--sample-token", type=str, required=True, help="nuScenes sample token")
    parser.add_argument("--out", type=str, default="coordiworld_scene_summary.json", help="Output JSON path")
    parser.add_argument("--full", action="store_true", help="Export full ICA SceneSummary instead of the CoordiWorld-oriented subset")
    parser.add_argument("--no-simulate-conflicts", action="store_true", help="Do not inject mild pseudo detector disagreement")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from nuscenes.nuscenes import NuScenes  # type: ignore

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    builder = NuScenesICAExampleBuilder(nusc=nusc, dataroot=args.dataroot)
    pipeline = ICAPipeline(config=ICAConfig(), map_extractor=NuScenesMapExtractor(args.dataroot))
    result = builder.run_pipeline(
        pipeline=pipeline,
        sample_token=args.sample_token,
        simulate_conflicts=not args.no_simulate_conflicts,
    )

    export = result.scene_summary.to_dict(coordiworld_view=not args.full)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"Saved SceneSummary to: {args.out}")
    print(json.dumps(export, ensure_ascii=False, indent=2)[:4000])
