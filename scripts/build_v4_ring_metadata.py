"""Materialize the selected V4 ring metadata from the audited CSV manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    rows = list(csv.DictReader(source.open("r", encoding="utf-8-sig", newline="")))
    records = []
    for row in rows:
        if str(row.get("selected_for_geometry", "")).strip().lower() != "true":
            continue
        records.append(
            {
                "relative_path": str(row["relative_path"]),
                "logical_ring_id": str(row["logical_ring_id"]),
                "frame_index_within_logical_ring": int(row["frame_index_within_logical_ring"]),
                "phase_01": float(row["phase_01"]),
                "selected_for_geometry": True,
            }
        )
    records.sort(key=lambda value: (value["logical_ring_id"], value["frame_index_within_logical_ring"], value["relative_path"]))
    payload = {
        "schema_version": 1,
        "method": "selected V4 source_manifest.csv rows; no new selection",
        "source_manifest_path": str(source),
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "record_count": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"record_count": len(records), "source_manifest_sha256": payload["source_manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
