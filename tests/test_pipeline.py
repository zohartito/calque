import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from calque.calibrate import Transform, fit_residuals, from_anchors, from_scalebar
from calque.export import to_dxf, to_svg
from calque.refine import refine_layers
from calque.vectorize import VectorizeParams, extract
from check_outputs import check_outputs, measured_segment
from synthetic_plan import make_plan, SCALE, WALL_PX


@pytest.fixture
def plan(tmp_path):
    return make_plan(tmp_path / "plan.png")


@pytest.mark.parametrize("explicit_refine", [True, False])
def test_public_pipeline(plan, tmp_path, explicit_refine):
    t = from_scalebar(200, 10)
    assert t.scale == pytest.approx(SCALE, abs=1e-12)
    layers, style = extract(str(plan), VectorizeParams(refine=not explicit_refine), return_style=True)
    assert all(layers[name] for name in ("heavy", "light", "green", "accent"))
    if explicit_refine:
        # Call the public refinement stage directly on the extracted strokes.
        layers.update(refine_layers({name: layers[name] for name in ("heavy", "light")}))
        style = None  # pre-refinement style arrays no longer match the polylines
    measured_segment(layers["heavy"], WALL_PX, tolerance=6)
    dxf, svg, calib = (tmp_path / name for name in ("plan.dxf", "plan.svg", "plan.json"))
    t.save(calib)
    assert Transform.load(calib).to_world(360, 60) == pytest.approx((18, -3))
    assert to_dxf(layers, t, dxf, style=style) > 0
    assert to_svg(layers, t, svg, style=style) > 0
    check_outputs(dxf, svg, calib, styled=not explicit_refine)


def test_anchor_calibration_roundtrip():
    # A rotated, translated 0.05 m/px drawing with analytically known points.
    pairs = [((0, 0), (2, 4)), ((100, 0), (6, 7)), ((0, 100), (5, 0))]
    t = from_anchors(pairs)
    assert t.scale == pytest.approx(SCALE, abs=1e-12)
    assert t.to_world(100, 100) == pytest.approx((9, 3), abs=1e-10)
    assert t.to_px(9, 3) == pytest.approx((100, 100), abs=1e-10)
    assert max(fit_residuals(t, pairs)) < 1e-10


def test_refinement_bridges_gap_within_its_pixel_space_limit():
    raw = {"heavy": [np.array([[100., 310.], [196., 310.]]),
                     np.array([[204., 310.], [300., 310.]])]}
    result = refine_layers(raw)
    assert len(result["heavy"]) == 1
    length, _ = measured_segment(result["heavy"], np.array([[100, 310], [300, 310]]), tolerance=1e-8)
    assert length == pytest.approx(200)


@pytest.mark.xfail(strict=True, reason="8 px raster gap becomes 12 px after thinning; default merge_gap_px is 8")
def test_default_refinement_bridges_raster_gap(plan):
    layers = extract(str(plan))
    length, _ = measured_segment(layers["heavy"], np.array([[100, 310], [300, 310]]), tolerance=6)
    assert length == pytest.approx(200, abs=6)


@pytest.mark.parametrize("calibration", ["scalebar", "scale", "anchors", "saved"])
def test_installed_cli_pipeline(plan, tmp_path, calibration):
    command = Path(sys.executable).with_name("calque")
    options = {
        "scalebar": ["--scalebar", "200=10"],
        "scale": ["--scale", "0.05"],
        "anchors": ["--anchor", "60,60=3,-3", "--anchor", "360,60=18,-3", "--anchor", "60,260=3,-13"],
        "saved": ["--calib", str(tmp_path / "saved.json")],
    }
    if calibration == "saved":
        (tmp_path / "saved.json").write_text(json.dumps({"a": SCALE, "b": 0, "tx": 0, "ty": 0}))
    dxf, svg, calib, preview = (tmp_path / name for name in ("plan.dxf", "plan.svg", "plan.json", "preview.png"))
    result = subprocess.run([str(command), "trace", str(plan), *options[calibration],
                             "--dxf", str(dxf), "--svg", str(svg),
                             "--save-calib", str(calib), "--preview", str(preview)],
                            text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    check_outputs(dxf, svg, calib)
    image = cv2.imread(str(preview))
    assert image is not None and image.shape == (480, 640, 3)
    assert np.count_nonzero(image.min(axis=2) < 200) > 1000
