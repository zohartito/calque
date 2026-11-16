"""Regressions for the confirmed public-repository audit findings."""
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import cv2
import ezdxf
import numpy as np
import pytest

from calque.calibrate import from_scale
from calque.export import to_dxf, to_svg
from calque.vectorize import VectorizeParams, extract
from synthetic_plan import make_plan


@pytest.fixture
def wall(tmp_path):
    image = np.full((100, 240, 3), 255, np.uint8)
    cv2.line(image, (20, 50), (220, 50), (0, 0, 0), 5)
    source = tmp_path / "wall.png"
    assert cv2.imwrite(str(source), image)
    return source


def run_trace(source, *options):
    return subprocess.run([str(Path(sys.executable).with_name("calque")),
                           "trace", str(source), "--scale", "0.05", *map(str, options)],
                          cwd=source.parent, capture_output=True, text=True)


@pytest.mark.parametrize("refine", [True, False])
@pytest.mark.parametrize("crop", [None, (10, 10, 590, 390)])
def test_cal_01_dense_pale_contours_have_style(tmp_path, refine, crop):
    image = np.full((400, 600, 3), 255, np.uint8)
    for y in range(20, 380, 10):
        cv2.line(image, (20, y), (580, y), (160, 160, 160), 3)
    cv2.line(image, (300, 10), (300, 390), (0, 0, 0), 3)
    source = tmp_path / "dense-pale.png"
    assert cv2.imwrite(str(source), image)
    layers, style = extract(str(source), VectorizeParams(refine=refine), crop,
                            return_style=True)
    for name, expected_tone in (("heavy", 0), ("light", 160)):
        assert layers[name]
        assert len(style["width"][name]) == len(style["tone"][name]) == len(layers[name])
        assert all(np.isfinite(w) and w > 0 for w in style["width"][name])
        assert np.median(style["tone"][name]) == pytest.approx(expected_tone)
    svg, dxf = tmp_path / "dense.svg", tmp_path / "dense.dxf"
    assert to_svg(layers, from_scale(0.05), svg, style=style) > 0
    assert to_dxf(layers, from_scale(0.05), dxf, style=style) > 0
    assert ET.parse(svg).find(".//{*}g[@id='calque-light']/{*}polyline") is not None
    assert ezdxf.readfile(dxf).modelspace().query('LWPOLYLINE[layer=="CALQUE_LIGHT"]')


@pytest.mark.parametrize("destination", ["missing/sub/preview.png", "preview.unknown"])
def test_cal_03_preview_write_failure_is_an_error(wall, tmp_path, destination):
    preview = tmp_path / destination
    result = run_trace(wall, "--preview", preview)
    assert result.returncode != 0
    assert "cannot write preview" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert "preview ->" not in result.stdout
    assert not preview.exists()


@pytest.mark.parametrize("vertical", [False, True])
@pytest.mark.parametrize("styled", [False, True])
def test_cal_02_svg_bounds_include_collinear_strokes(tmp_path, vertical, styled):
    points = np.array([[20., 50.], [220., 50.]])
    if vertical:
        points = points[:, ::-1]
    style = {"width": {"heavy": [5.]}, "tone": {"heavy": [0.]}} if styled else None
    svg = tmp_path / "wall.svg"
    assert to_svg({"heavy": [points]}, from_scale(0.05), svg, style) == 1
    root = ET.parse(svg).getroot()
    x, y, width, height = map(float, root.attrib["viewBox"].split())
    assert width > 0 and height > 0
    assert float(root.attrib["width"][:-2]) == width
    assert float(root.attrib["height"][:-2]) == height
    line = root.find(".//{*}polyline")
    actual = np.array([list(map(float, pair.split(",")))
                       for pair in line.attrib["points"].split()])
    half_stroke = (250. if styled else 0.35) / 2
    assert np.all(actual.min(axis=0) - half_stroke >= [x, y])
    assert np.all(actual.max(axis=0) + half_stroke <= [x + width, y + height])
    assert np.linalg.norm(actual[1] - actual[0]) == pytest.approx(10000.)


def test_cal_02_horizontal_raster_exports_positive_svg(wall, tmp_path):
    svg = tmp_path / "wall.svg"
    result = run_trace(wall, "--svg", svg)
    assert result.returncode == 0, result.stderr
    root = ET.parse(svg).getroot()
    assert all(float(v) > 0 for v in root.attrib["viewBox"].split()[2:])


OUTPUTS = ("--save-calib", "--dxf", "--svg", "--preview")


@pytest.mark.parametrize("option", OUTPUTS)
@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_cal_05_rejects_source_aliases(wall, tmp_path, option, alias_kind):
    original = wall.read_bytes()
    target = wall
    if alias_kind != "direct":
        target = tmp_path / "alias.png"
        if alias_kind == "symlink":
            target.symlink_to(wall)
        else:
            target.hardlink_to(wall)
    result = run_trace(wall, option, target)
    assert result.returncode == 2
    assert "aliases input" in result.stderr
    assert wall.read_bytes() == original


@pytest.mark.parametrize("first,second", list(combinations(OUTPUTS, 2)))
def test_cal_05_rejects_duplicate_outputs(wall, tmp_path, first, second):
    target = tmp_path / "shared.png"
    result = run_trace(wall, first, target, second, target)
    assert result.returncode == 2
    assert "duplicate output" in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("first,second", list(combinations(OUTPUTS, 2)))
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("linked_parent", [False, True])
def test_cal_05_new_case_variant_outputs_follow_filesystem(
        wall, tmp_path, first, second, overwrite, linked_parent):
    original = wall.read_bytes()
    case_insensitive = wall.with_name("WALL.PNG").exists()
    target = tmp_path / "shared.png"
    parent = tmp_path
    if linked_parent:
        parent = tmp_path / "linked"
        parent.symlink_to(tmp_path, target_is_directory=True)
    alias = parent / "SHARED.PNG"
    before = set(tmp_path.iterdir())
    assert not target.exists() and not alias.exists()
    result = run_trace(wall, first, target, second, alias,
                       *(["--overwrite"] if overwrite else []))
    if case_insensitive:
        assert result.returncode == 2, result.stdout + result.stderr
        assert "duplicate output" in result.stderr
        assert set(tmp_path.iterdir()) == before
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert target.is_file() and alias.is_file()
        assert not target.samefile(alias)
    assert wall.read_bytes() == original


def test_cal_05_case_variant_outputs_in_distinct_directories(wall, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    calibration, preview = first / "shared.png", second / "SHARED.PNG"
    result = run_trace(wall, "--save-calib", calibration, "--preview", preview)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(calibration.read_text())["a"] == pytest.approx(0.05)
    assert cv2.imread(str(preview)) is not None
    assert not calibration.samefile(preview)


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_cal_05_overwrite_cannot_allow_linked_duplicates(wall, tmp_path, alias_kind):
    target = tmp_path / "shared.png"
    target.write_bytes(b"keep this output")
    alias = tmp_path / "alias.png"
    if alias_kind == "symlink":
        alias.symlink_to(target)
    else:
        alias.hardlink_to(target)
    result = run_trace(wall, "--svg", target, "--preview", alias, "--overwrite")
    assert result.returncode == 2
    assert "duplicate output" in result.stderr
    assert target.read_bytes() == alias.read_bytes() == b"keep this output"


@pytest.mark.parametrize("option", OUTPUTS)
@pytest.mark.parametrize("overwrite", [False, True])
def test_cal_05_existing_output_requires_consent(wall, tmp_path, option, overwrite):
    target = tmp_path / "existing.png"
    target.write_bytes(b"keep this output")
    result = run_trace(wall, option, target, *(["--overwrite"] if overwrite else []))
    if overwrite:
        assert result.returncode == 0, result.stderr
        assert target.read_bytes() != b"keep this output"
    else:
        assert result.returncode == 2
        assert "--overwrite" in result.stderr
        assert target.read_bytes() == b"keep this output"


def test_cal_05_overwrite_cannot_allow_input_alias_or_partial_writes(wall, tmp_path):
    original = wall.read_bytes()
    calibration = tmp_path / "new.json"
    result = run_trace(wall, "--save-calib", calibration, "--preview", wall, "--overwrite")
    assert result.returncode == 2
    assert "aliases input" in result.stderr
    assert wall.read_bytes() == original
    assert not calibration.exists()


def test_cal_05_protects_loaded_calibration(wall, tmp_path):
    calibration = tmp_path / "input.json"
    from_scale(0.05).save(calibration)
    original = calibration.read_bytes()
    result = run_trace(wall, "--calib", calibration, "--svg", calibration)
    assert result.returncode == 2
    assert "aliases input" in result.stderr
    assert calibration.read_bytes() == original


def test_cal_06_evaluation_note_discloses_ungated_fit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    image = make_plan(tmp_path / "plan.png")
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"theta_deg": 0, "scale": 0.05,
                                       "crater_px": [0, 0], "crater_m": [56.64, 31.96]}))
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"osm": [{"id": 1064543187, "pts": [
        [59.64, 28.96], [74.64, 28.96], [74.64, 18.96], [59.64, 18.96],
        [77.64, 28.96], [83.64, 22.96]]}]}))
    output = tmp_path / "results.json"
    result = subprocess.run([sys.executable, str(root / "evals/eval_lvv.py"),
                             str(image), str(geometry), str(registration), "--output", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    note = json.loads(output.read_text())["note"]
    assert "correction magnitude is not gated" in note
    assert "must stay near identity" not in note
    assert json.loads((root / "evals/results_lvv.json").read_text())["note"] == note
