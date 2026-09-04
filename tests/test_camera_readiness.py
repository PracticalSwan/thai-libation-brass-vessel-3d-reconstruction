from pathlib import Path

from PIL import Image

from camera_readiness import (
    CameraRecord,
    camera_signature,
    read_camera_record,
    summarize_camera_records,
)


def _record(index: int, *, focal: float | None = 3.98, zoom: float | None = 1.0):
    return CameraRecord(
        selected_index=index,
        filename=f"frame_{index:03d}.jpg",
        width=3072,
        height=4080,
        orientation=1,
        make="OPPO",
        model="OPPO Reno12 F",
        lens_model="OPPO Reno12 F back camera 26mm f/1.8",
        focal_length_mm=focal,
        focal_length_35mm=26.0 if focal is not None else None,
        digital_zoom_ratio=zoom,
        datetime_original=f"2026:08:26 12:00:{index:02d}",
    )


def test_read_camera_record_handles_missing_exif_without_guessing(tmp_path: Path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (12, 16), "white").save(path, format="JPEG")

    record = read_camera_record(1, path)

    assert record.width == 12
    assert record.height == 16
    assert record.make is None
    assert record.model is None
    assert record.focal_length_mm is None
    assert record.digital_zoom_ratio is None


def test_read_camera_record_reads_available_exif_fields(tmp_path: Path):
    path = tmp_path / "camera.jpg"
    exif = Image.Exif()
    exif[271] = "OPPO"
    exif[272] = "OPPO Reno12 F"
    exif[274] = 1
    exif[37386] = 3.98
    exif[41989] = 26
    exif[41988] = 1.0
    exif[42036] = "OPPO Reno12 F back camera 26mm f/1.8"
    exif[36867] = "2026:08:26 12:29:49"
    Image.new("RGB", (12, 16), "white").save(path, format="JPEG", exif=exif)

    record = read_camera_record(7, path)

    assert record.make == "OPPO"
    assert record.model == "OPPO Reno12 F"
    assert record.orientation == 1
    assert record.focal_length_mm == 3.98
    assert record.focal_length_35mm == 26.0
    assert record.digital_zoom_ratio == 1.0
    assert record.lens_model == "OPPO Reno12 F back camera 26mm f/1.8"


def test_same_signature_recommends_shared_intrinsics():
    records = (_record(1), _record(2), _record(3))

    summary = summarize_camera_records(records)

    assert summary["unique_signature_count"] == 1
    assert summary["camera_group_recommendation"] == "shared_intrinsics_single_camera_signature"
    assert summary["signature_groups"][0]["frame_count"] == 3


def test_different_focal_or_zoom_requires_separate_signature_groups():
    records = (_record(1), _record(2, focal=4.2), _record(3, zoom=1.2))

    summary = summarize_camera_records(records)

    assert summary["unique_signature_count"] == 3
    assert summary["camera_group_recommendation"] == "separate_intrinsics_by_camera_signature"


def test_missing_focal_metadata_is_counted_not_invented():
    records = (_record(1, focal=None), _record(2, focal=None))

    summary = summarize_camera_records(records)

    assert summary["missing_fields"]["focal_length_mm"] == 2
    assert summary["missing_fields"]["focal_length_35mm"] == 2
    assert camera_signature(records[0])[6] is None
