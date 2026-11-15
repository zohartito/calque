"""Parse exported files and check independent, known synthetic dimensions."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import ezdxf
import numpy as np

from synthetic_plan import DIAGONAL_PX, LENGTH_TOLERANCE_M, SCALE, WALL_METERS, WALL_PX

NS = {"svg": "http://www.w3.org/2000/svg"}
EXPECTED = {"CALQUE_HEAVY", "CALQUE_LIGHT", "CALQUE_VEG", "CALQUE_ACCENT"}


def measured_segment(polylines, endpoints, tolerance=LENGTH_TOLERANCE_M):
    """Find the actual wall by both endpoints, not just a coincidental length."""
    for poly in polylines:
        for a, b in zip(poly, poly[1:]):
            direct = max(np.linalg.norm(a - endpoints[0]), np.linalg.norm(b - endpoints[1]))
            reverse = max(np.linalg.norm(b - endpoints[0]), np.linalg.norm(a - endpoints[1]))
            if min(direct, reverse) <= tolerance:
                return float(np.linalg.norm(b - a)), np.asarray(b) - a
    raise AssertionError(f"no segment matches expected endpoints {endpoints.tolist()}")


def check_outputs(dxf_path, svg_path, calib_path, *, styled=True):
    for path in (dxf_path, svg_path, calib_path):
        assert Path(path).is_file() and Path(path).stat().st_size > 0
    calib = json.loads(Path(calib_path).read_text())
    assert math.isclose(calib["scale_m_per_px"], SCALE, abs_tol=1e-12)
    assert math.isclose(calib["a"], SCALE, abs_tol=1e-12)
    assert abs(calib["b"]) < 1e-12
    assert abs(calib["tx"]) < 1e-10 and abs(calib["ty"]) < 1e-10
    doc = ezdxf.readfile(dxf_path)
    assert doc.units == ezdxf.units.M
    assert not doc.audit().has_errors
    entities = list(doc.modelspace())
    layers = {entity.dxf.layer for entity in entities}
    assert EXPECTED <= layers  # entities must exist on the layers, not only table entries
    if styled:
        assert {"CALQUE_VEG_FILL", "CALQUE_ACCENT_FILL"} <= layers
        assert all(entity.dxftype() == "HATCH" for entity in entities
                   if entity.dxf.layer.endswith("_FILL"))
    heavy = [np.array(entity.get_points("xy")) for entity in entities
             if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == "CALQUE_HEAVY"]
    # Expected world coordinates are computed from fixture dimensions, independently
    # of the calibration and export functions under test.
    wall, _ = measured_segment(heavy, WALL_PX * [SCALE, -SCALE])
    assert abs(wall - WALL_METERS) <= LENGTH_TOLERANCE_M
    diagonal, direction = measured_segment(heavy, DIAGONAL_PX * [SCALE, -SCALE])
    assert abs(diagonal - math.sqrt(2) * 120 * SCALE) <= LENGTH_TOLERANCE_M
    angle = math.degrees(math.atan2(direction[1], direction[0])) % 180
    assert abs(angle - 135) <= 2  # the diagonal must survive orthogonal snapping

    root = ET.parse(svg_path).getroot()
    assert root.tag == f"{{{NS['svg']}}}svg"
    assert root.attrib["width"].endswith("mm") and root.attrib["height"].endswith("mm")
    assert all(float(v) > 0 for v in root.attrib["viewBox"].split()[2:])
    groups = {group.attrib["id"]: group for group in root.findall("svg:g", NS)}
    assert {"calque-heavy", "calque-light", "calque-green", "calque-accent"} <= groups.keys()
    assert all(len(groups[name]) > 0 for name in ("calque-heavy", "calque-light", "calque-green", "calque-accent"))
    svg_heavy = [np.array([[float(v) for v in pair.split(",")]
                          for pair in line.attrib["points"].split()])
                 for line in groups["calque-heavy"].findall("svg:polyline", NS)]
    # SVG is translated to its bounds and y-down; compare its independently parsed
    # wall with the DXF wall after the specified meter-to-mm conversion.
    all_xy = [np.array(e.get_points("xy")) for e in entities if e.dxftype() == "LWPOLYLINE"]
    points = np.vstack(all_xy)
    origin = np.array([points[:, 0].min(), points[:, 1].max()])
    svg_wall_endpoints = (WALL_PX * [SCALE, -SCALE] - origin) * [1000, -1000]
    svg_wall, _ = measured_segment(svg_heavy, svg_wall_endpoints, LENGTH_TOLERANCE_M * 1000)
    assert abs(svg_wall / 1000 - WALL_METERS) <= LENGTH_TOLERANCE_M
    assert abs(svg_wall / 1000 - wall) < 0.0001
    return {"scale_m_per_px": calib["scale_m_per_px"], "known_scale_m_per_px": SCALE,
            "wall_m": wall, "known_wall_m": WALL_METERS,
            "wall_tolerance_m": LENGTH_TOLERANCE_M, "diagonal_m": diagonal,
            "svg_wall_mm": svg_wall,
            "dxf_entities": len(entities), "dxf_layers": sorted(layers),
            "svg_groups": sorted(groups)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxf", type=Path)
    parser.add_argument("svg", type=Path)
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_outputs(args.dxf, args.svg, args.calibration), indent=2))
