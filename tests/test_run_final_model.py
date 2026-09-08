from pathlib import Path
import json

import pytest

import run_final_model as runner


def test_final_stage_order_starts_with_cv_before_blender():
    assert runner.STAGE_ORDER[:4] == (
        "analyze",
        "cv-fit",
        "base",
        "geometry-validate",
    )
    assert runner.STAGE_ORDER[-2:] == ("final-validate", "export")


def test_require_accepted_report_fails_closed(tmp_path: Path):
    report = tmp_path / "final_cv_fit.json"

    with pytest.raises(ValueError, match="missing required report"):
        runner.require_accepted_report(report)

    report.write_text('{"accepted": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not accepted"):
        runner.require_accepted_report(report)


def test_run_all_stops_before_export_for_user_review(monkeypatch):
    observed = []
    monkeypatch.setattr(
        runner,
        "run_stage",
        lambda stage, project_root=runner.PROJECT_ROOT: observed.append(stage) or {"stage": stage},
    )

    result = runner.run_all_through()
    expected = list(runner.STAGE_ORDER[: runner.STAGE_ORDER.index("final-validate") + 1])

    assert observed == expected
    assert result["completed_stages"] == expected
    assert result["accepted"] is False
    assert result["user_review_required_before_export"] is True


def test_run_all_stops_at_requested_stage(monkeypatch):
    observed = []
    monkeypatch.setattr(
        runner,
        "run_stage",
        lambda stage, project_root=runner.PROJECT_ROOT: observed.append(stage) or {"stage": stage},
    )

    runner.run_all_through("cleanup")

    assert observed == list(runner.STAGE_ORDER[: runner.STAGE_ORDER.index("cleanup") + 1])


def test_cv_fit_requires_separate_accepted_visual_review(monkeypatch, tmp_path: Path):
    report_dir = tmp_path / runner.V2_RELATIVE_ROOT / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "final_reference_evidence.json").write_text(
        '{"accepted": true}\n', encoding="utf-8"
    )
    observed = {}

    class Result:
        accepted = False
        metrics_passed = True
        aggregate_metrics = {"median_silhouette_iou": 0.91}
        visual_component_review = {}

    def fake_fit(project_root, *, visual_component_review):
        observed["root"] = project_root
        observed["review"] = visual_component_review
        return Result()

    def fake_write(project_root, result):
        report_path = report_dir / "final_cv_fit.json"
        profiles_path = report_dir / "final_profiles.json"
        report_path.write_text(
            json.dumps({"candidate_sha256": "candidate"}), encoding="utf-8"
        )
        profiles_path.write_text("{}", encoding="utf-8")
        return report_path, profiles_path

    def fake_replace(result, **changes):
        replacement = Result()
        for key, value in changes.items():
            setattr(replacement, key, value)
        return replacement

    monkeypatch.setattr("final_cv_model_fit.fit_vessel_model", fake_fit)
    monkeypatch.setattr("final_cv_model_fit.write_fit_outputs", fake_write)
    monkeypatch.setattr(runner, "replace", fake_replace)

    with pytest.raises(ValueError, match="cv_visual_review.json"):
        runner._run_cv_fit(tmp_path)

    review = {
        "accepted": True,
        "candidate_sha256": "candidate",
        "components": {
            "bowl": "pass",
            "globe": "pass",
            "neck": "pass",
            "lid": "pass",
            "finial": "pass",
        },
    }
    (report_dir / "cv_visual_review.json").write_text(
        json.dumps(review), encoding="utf-8"
    )

    result = runner._run_cv_fit(tmp_path)

    assert observed["root"] == tmp_path
    assert observed["review"] is None
    assert result["accepted"] is True
    assert result["metrics_passed"] is True


def test_export_requires_explicit_user_approval(tmp_path: Path):
    v2_root = tmp_path / runner.V2_RELATIVE_ROOT
    report_dir = v2_root / "reports"
    report_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="user approves"):
        runner.require_user_export_approval(v2_root)

    approval = report_dir / runner.USER_EXPORT_APPROVAL_REPORT
    approval.write_text('{"approved": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="approval is not true"):
        runner.require_user_export_approval(v2_root)

    approval.write_text('{"approved": true}\n', encoding="utf-8")
    assert runner.require_user_export_approval(v2_root)["approved"] is True
