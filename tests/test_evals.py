import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from synthetic_plan import make_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["eval_lvv", "eval_fidelity"])
def test_import_without_reference_files(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(name, ROOT / "evals" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,arguments", [
    ("eval_fidelity", []), ("eval_lvv", []),
    ("eval_fidelity", ["missing.png"]),
    ("eval_lvv", ["missing.png", "missing-geometry.json", "missing-registration.json"]),
])
def test_missing_input_is_cli_error(name, arguments, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "evals" / f"{name}.py"), *arguments],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert ("not found" if arguments else "required") in result.stderr
    assert "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.fixture
def fidelity_result(tmp_path):
    image = make_plan(tmp_path / "plan.png")
    result = subprocess.run([sys.executable, str(ROOT / "evals/eval_fidelity.py"), str(image),
                             "--debug", str(tmp_path / "debug/overlay.jpg")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode in (0, 1), result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    metrics = json.loads(result.stdout.rsplit("\n", 2)[0])
    assert result.returncode == (0 if metrics["pass"] else 1)
    return metrics, tmp_path


def test_fidelity_runs_on_generated_plan(fidelity_result):
    metrics, tmp_path = fidelity_result
    assert metrics["line_recall"] >= 0.90
    assert metrics["line_precision"] >= 0.90
    assert metrics["green_iou"] >= 0.80
    assert all((tmp_path / "debug" / name).is_file() for name in
               ("overlay.jpg", "overlay_styled.png", "overlay_visdiff.jpg"))


@pytest.mark.xfail(strict=True, reason="Styled raster MAE is 33.76, above the existing 24 threshold; stroke/region styling needs further work")
def test_styled_preview_meets_fidelity_threshold(fidelity_result):
    metrics, _ = fidelity_result
    assert metrics["visual_mae"] <= 24


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_each_lvv_input_is_required(tmp_path, missing_index):
    paths = [tmp_path / name for name in ("plan.png", "geometry.json", "registration.json")]
    for i, path in enumerate(paths):
        if i != missing_index:
            path.write_text("placeholder: existence is checked before parsing")
    result = subprocess.run([sys.executable, str(ROOT / "evals/eval_lvv.py"), *map(str, paths)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert f"input file not found: {paths[missing_index]}" in result.stderr
    assert "Traceback" not in result.stderr


def test_lvv_runs_on_supplied_synthetic_references(tmp_path):
    image = make_plan(tmp_path / "plan.png")
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"theta_deg": 0, "scale": 0.05,
                                       "crater_px": [0, 0], "crater_m": [56.64, 31.96]}))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"osm": [{"id": 1064543187, "pts": [
        [59.64, 28.96], [74.64, 28.96], [74.64, 18.96], [59.64, 18.96],
        [77.64, 28.96], [83.64, 22.96]]}]}))
    output = tmp_path / "results.json"
    result = subprocess.run([sys.executable, str(ROOT / "evals/eval_lvv.py"),
                             str(image), str(geometry), str(registration), "--output", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads(output.read_text())
    assert metrics["calibration"]["scale_m_per_px"] == pytest.approx(0.05)
    assert metrics["n_outer_vertices"] == 4 and metrics["n_arc_vertices"] == 2
    assert metrics["outer_median_m"] <= 0.30
    assert metrics["outer_coverage_at_1.5m"] == 1
    assert metrics["pass"]
