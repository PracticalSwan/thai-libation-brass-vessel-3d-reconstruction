"""Measure whether audited two-view inliers remain inside immutable vessel masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v4_repair import create_disposable_sqlite_from_manifest, load_canonical_sqlite_snapshot_manifest


def _mask_fraction(points: np.ndarray, mask: np.ndarray) -> float:
    values = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int64)
    height, width = mask.shape[:2]
    in_bounds = (
        (values[:, 0] >= 0)
        & (values[:, 0] < width)
        & (values[:, 1] >= 0)
        & (values[:, 1] < height)
    )
    supported = np.zeros(len(values), dtype=bool)
    if in_bounds.any():
        supported[in_bounds] = mask[values[in_bounds, 1], values[in_bounds, 0]] > 0
    return float(supported.mean()) if len(values) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, help="Deprecated compatibility check; never opened.")
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot_manifest = load_canonical_sqlite_snapshot_manifest(args.snapshot_manifest)
    canonical_path = Path(snapshot_manifest["canonical_path"]).resolve()
    if args.database is not None and args.database.resolve() != canonical_path:
        raise ValueError("--database does not match --snapshot-manifest canonical_path")
    pair_payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = pair_payload.get("pairs") if isinstance(pair_payload, dict) else pair_payload
    if not isinstance(pairs, list):
        raise ValueError("pairs JSON must contain a pairs list")
    with tempfile.TemporaryDirectory(prefix="v4_match_mask_audit_") as temporary:
        working = Path(temporary) / "working.db"
        disposable = create_disposable_sqlite_from_manifest(args.snapshot_manifest, working)
        import pycolmap

        database = pycolmap.Database.open(str(working))
        try:
            images = {str(image.name): image for image in database.read_all_images()}
            records = []
            for pair in pairs:
                first_name, second_name = str(pair["first"]), str(pair["second"])
                first, second = images[first_name], images[second_name]
                geometry = database.read_two_view_geometry(int(first.image_id), int(second.image_id))
                if geometry is None:
                    records.append({"first": first_name, "second": second_name, "status": "missing_geometry"})
                    continue
                matches = np.asarray(geometry.inlier_matches, dtype=np.int64).reshape(-1, 2)
                first_keypoints = np.asarray(database.read_keypoints(int(first.image_id)), dtype=np.float64)[:, :2]
                second_keypoints = np.asarray(database.read_keypoints(int(second.image_id)), dtype=np.float64)[:, :2]
                valid = (
                    (matches[:, 0] >= 0)
                    & (matches[:, 0] < len(first_keypoints))
                    & (matches[:, 1] >= 0)
                    & (matches[:, 1] < len(second_keypoints))
                ) if len(matches) else np.empty(0, dtype=bool)
                matches = matches[valid]
                first_points = first_keypoints[matches[:, 0]] if len(matches) else np.empty((0, 2))
                second_points = second_keypoints[matches[:, 1]] if len(matches) else np.empty((0, 2))
                first_mask = cv2.imread(str(args.mask_root / first_name), cv2.IMREAD_GRAYSCALE)
                second_mask = cv2.imread(str(args.mask_root / second_name), cv2.IMREAD_GRAYSCALE)
                if first_mask is None or second_mask is None:
                    raise FileNotFoundError(f"missing feature mask for {first_name} or {second_name}")
                first_fraction = _mask_fraction(first_points, first_mask)
                second_fraction = _mask_fraction(second_points, second_mask)
                records.append({
                    "first": first_name,
                    "second": second_name,
                    "pair_id": int(min(first.image_id, second.image_id) * (2**31 - 1) + max(first.image_id, second.image_id)),
                    "verified_inliers": int(len(matches)),
                    "first_mask_support_fraction": first_fraction,
                    "second_mask_support_fraction": second_fraction,
                    "both_mask_support_fraction": float(min(first_fraction, second_fraction)),
                    "mask_supported": bool(first_fraction >= 0.95 and second_fraction >= 0.95),
                })
        finally:
            database.close()
    payload = {
        "schema_version": 1,
        "method": "inlier keypoint support against immutable feature masks via disposable SQLite copy",
        "database": str(canonical_path),
        "snapshot_manifest": str(args.snapshot_manifest.resolve()),
        "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "database_sha256": snapshot_manifest["canonical_sha256"],
        "mask_root": str(args.mask_root.resolve()),
        "pair_source_sha256": __import__("hashlib").sha256(args.pairs.read_bytes()).hexdigest(),
        "pair_count": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pair_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
