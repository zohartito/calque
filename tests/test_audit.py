"""Regressions for the confirmed public-repository audit findings."""
import xml.etree.ElementTree as ET

import cv2
import ezdxf
import numpy as np
import pytest

from calque.calibrate import from_scale
from calque.export import to_dxf, to_svg
from calque.vectorize import VectorizeParams, extract


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
