"""Regressions for the confirmed public-repository audit findings."""
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
