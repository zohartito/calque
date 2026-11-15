"""Write extracted layers to DXF (world meters, Rhino-ready) and SVG,
plus a styled raster preview that mimics the source drawing's look."""
from __future__ import annotations

from pathlib import Path

import cv2
import ezdxf
import numpy as np
import svgwrite

from .calibrate import Transform

LINE_LAYERS = ("light", "heavy")          # draw order: pale under dark
REGION_LAYERS = ("shade_light", "shade_dark", "green", "accent")
SHADE_LAYERS = ("shade_light", "shade_dark")   # fill only, no drawn outline


def render_styled(layers: dict[str, list[np.ndarray]], style: dict,
                  shape: tuple[int, int]) -> np.ndarray:
    """Raster preview in pixel space: fills, per-stroke widths and tones."""
    canvas = np.full((shape[0], shape[1], 3), 255, np.uint8)
    for name in REGION_LAYERS:
        rings = layers.get(name, [])
        holes = style.get("hole", {}).get(name, [False] * len(rings))
        fill = style.get("fill", {}).get(name, (128, 200, 128))
        outline = tuple(int(c * 0.55) for c in fill)
        for pass_holes in (False, True):     # painter: outers, then holes
            for ring, is_hole in zip(rings, holes):
                if is_hole != pass_holes:
                    continue
                pts = ring.astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(canvas, [pts], (255, 255, 255) if is_hole else fill)
        if name in SHADE_LAYERS:
            continue
        for ring, is_hole in zip(rings, holes):
            cv2.polylines(canvas, [ring.astype(np.int32).reshape(-1, 1, 2)],
                          True, outline, 1, cv2.LINE_AA)
    for cx, cy, w, h, ang in style.get("pots", []):
        box = ((cx, cy), (w, h), ang)
        cv2.ellipse(canvas, box, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, box, (40, 90, 40), 1, cv2.LINE_AA)
    for name in LINE_LAYERS:
        polys = layers.get(name, [])
        widths = style.get("width", {}).get(name, [1.5] * len(polys))
        tones = style.get("tone", {}).get(name, [40.0] * len(polys))
        for poly, w, t in zip(polys, widths, tones):
            g = int(np.clip(t, 0, 255))
            cv2.polylines(canvas, [poly.astype(np.int32).reshape(-1, 1, 2)],
                          False, (g, g, g), max(1, round(w)), cv2.LINE_AA)
    return canvas


def render_line(layers: dict[str, list[np.ndarray]], style: dict,
                shape: tuple[int, int]) -> np.ndarray:
    """Clean line-drawing preview: no fills, no shading -- weighted strokes,
    smooth region outlines, pot ellipses. What the DXF/SVG linework holds."""
    canvas = np.full((shape[0], shape[1], 3), 255, np.uint8)
    outline_of = {"green": (40, 110, 40), "accent": (0, 120, 200)}
    for name in ("green", "accent"):
        rings = layers.get(name, [])
        ws = style.get("width", {}).get(name, [1.0] * len(rings))
        for ring, w in zip(rings, ws):
            cv2.polylines(canvas, [ring.astype(np.int32).reshape(-1, 1, 2)],
                          True, outline_of[name], max(1, round(w)), cv2.LINE_AA)
    for cx, cy, w, h, ang in style.get("pots", []):
        cv2.ellipse(canvas, ((cx, cy), (w, h), ang), (40, 110, 40), 1, cv2.LINE_AA)
    for name in LINE_LAYERS:
        polys = layers.get(name, [])
        widths = style.get("width", {}).get(name, [1.5] * len(polys))
        tones = style.get("tone", {}).get(name, [40.0] * len(polys))
        for poly, w, t in zip(polys, widths, tones):
            g = int(np.clip(t, 0, 220))
            cv2.polylines(canvas, [poly.astype(np.int32).reshape(-1, 1, 2)],
                          False, (g, g, g), max(1, round(w)), cv2.LINE_AA)
    return canvas

DXF_LAYERS = {
    "heavy": ("CALQUE_HEAVY", 7),    # white/black
    "light": ("CALQUE_LIGHT", 8),    # gray
    "green": ("CALQUE_VEG", 3),      # green
    "pots": ("CALQUE_POTS", 3),      # planting pots as true ellipses
    "accent": ("CALQUE_ACCENT", 30), # orange: colored fills (diagram planes)
    "shade_dark": ("CALQUE_SHADE_DK", 8),   # posterized soft shading
    "shade_light": ("CALQUE_SHADE_LT", 9),
}
SVG_COLORS = {"heavy": "#111111", "light": "#888888", "green": "#3a8f3a", "accent": "#e08a00",
              "shade_dark": "#b5b5b5", "shade_light": "#d8d8d8"}


_DXF_LW = [13, 18, 25, 35, 50, 70, 100]     # hundredths of mm


def _lineweight(width_px: float) -> int:
    """Map a measured pixel stroke width to the nearest DXF plot weight."""
    target = width_px * 13                   # 1px stroke ~ 0.13mm pen
    return min(_DXF_LW, key=lambda w: abs(w - target))


def to_dxf(layers: dict[str, list[np.ndarray]], transform: Transform,
           path: str | Path, style: dict | None = None) -> int:
    doc = ezdxf.new("R2018", setup=True)
    doc.units = ezdxf.units.M
    msp = doc.modelspace()
    for name, (dxf_layer, color) in DXF_LAYERS.items():
        if name not in layers:
            continue
        doc.layers.add(dxf_layer, color=color)
    n = 0
    for name, polys in layers.items():
        if name == "pots" and style and style.get("pots"):
            continue                    # emitted as true ELLIPSE entities below
        dxf_layer = DXF_LAYERS.get(name, ("CALQUE_MISC", 7))[0]
        widths = (style or {}).get("width", {}).get(name)
        for i, poly in enumerate(polys):
            world = [transform.to_world(float(x), float(y)) for x, y in poly]
            closed = (abs(world[0][0] - world[-1][0]) < 1e-9 and
                      abs(world[0][1] - world[-1][1]) < 1e-9)
            pts = world[:-1] if closed else world
            if len(pts) < 2:
                continue
            attribs = {"layer": dxf_layer}
            if widths is not None and i < len(widths):
                attribs["lineweight"] = _lineweight(widths[i])
            msp.add_lwpolyline(pts, close=closed, dxfattribs=attribs)
            n += 1
    if style and style.get("pots"):
        if "CALQUE_POTS" not in doc.layers:
            doc.layers.add("CALQUE_POTS", color=3)
        for cx, cy, w, h, ang in style["pots"]:
            if w >= h:
                axis_deg, ratio, major = ang, h / w, w / 2.0
            else:
                axis_deg, ratio, major = ang + 90.0, w / h, h / 2.0
            ex = cx + major * np.cos(np.radians(axis_deg))
            ey = cy + major * np.sin(np.radians(axis_deg))
            c_w = transform.to_world(float(cx), float(cy))
            e_w = transform.to_world(float(ex), float(ey))
            msp.add_ellipse(center=c_w,
                            major_axis=(e_w[0] - c_w[0], e_w[1] - c_w[1]),
                            ratio=min(ratio, 1.0),
                            dxfattribs={"layer": "CALQUE_POTS"})
            n += 1
    if style:   # solid vegetation/accent fills as hatches under the linework
        for name in REGION_LAYERS:
            rings = layers.get(name, [])
            if not rings:
                continue
            holes = style["hole"].get(name, [False] * len(rings))
            parents = style["parent"].get(name, [-1] * len(rings))
            b, g, r = style["fill"].get(name, (128, 200, 128))
            fill_layer = DXF_LAYERS.get(name, ("CALQUE_MISC", 7))[0] + "_FILL"
            doc.layers.add(fill_layer)
            for i, ring in enumerate(rings):
                if holes[i]:
                    continue
                hatch = msp.add_hatch(dxfattribs={"layer": fill_layer})
                hatch.rgb = (r, g, b)
                world = [transform.to_world(float(x), float(y)) for x, y in ring[:-1]]
                hatch.paths.add_polyline_path(world, is_closed=True,
                                              flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL)
                for j, hole in enumerate(rings):
                    if holes[j] and parents[j] == i:
                        hw = [transform.to_world(float(x), float(y)) for x, y in hole[:-1]]
                        hatch.paths.add_polyline_path(hw, is_closed=True,
                                                      flags=ezdxf.const.BOUNDARY_PATH_OUTERMOST)
                n += 1
    doc.saveas(str(path))
    return n


def to_svg(layers: dict[str, list[np.ndarray]], transform: Transform,
           path: str | Path, style: dict | None = None) -> int:
    """SVG in millimeters, world-scaled, y flipped to match world north-up.

    With a style dict, regions become filled paths (holes carved via evenodd)
    and lines carry their measured stroke width and tone, so the SVG opens in
    Illustrator looking like the source drawing.
    """
    all_pts = [transform.to_world(float(x), float(y))
               for polys in layers.values() for poly in polys for x, y in poly]
    if not all_pts:
        raise ValueError("nothing to export")
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w_mm, h_mm = (x1 - x0) * 1000, (y1 - y0) * 1000
    mm_per_px = transform.scale * 1000 if hasattr(transform, "scale") else 1.0
    dwg = svgwrite.Drawing(str(path), size=(f"{w_mm:.1f}mm", f"{h_mm:.1f}mm"),
                           viewBox=f"0 0 {w_mm:.1f} {h_mm:.1f}")

    def mm(poly):
        return [(round((X - x0) * 1000, 2), round((y1 - Y) * 1000, 2))
                for X, Y in (transform.to_world(float(x), float(y)) for x, y in poly)]

    n = 0
    if style is None:
        for name, polys in layers.items():
            grp = dwg.g(id=f"calque-{name}", stroke=SVG_COLORS.get(name, "#000"),
                        fill="none", stroke_width=0.35)
            for poly in polys:
                grp.add(dwg.polyline(points=mm(poly)))
                n += 1
            dwg.add(grp)
        dwg.save()
        return n

    rgb = lambda b, g, r: f"#{r:02x}{g:02x}{b:02x}"
    for name in REGION_LAYERS:          # fills first, linework on top
        rings = layers.get(name, [])
        if not rings:
            continue
        holes = style["hole"].get(name, [False] * len(rings))
        parents = style["parent"].get(name, [-1] * len(rings))
        fill = rgb(*style["fill"].get(name, (128, 200, 128)))
        outline = rgb(*(int(c * 0.55) for c in style["fill"].get(name, (128, 200, 128))))
        grp = dwg.g(id=f"calque-{name}", fill=fill, fill_rule="evenodd",
                    stroke="none" if name in SHADE_LAYERS else outline,
                    stroke_width=round(mm_per_px, 2))
        ring_w = style.get("width", {}).get(name)
        for i, ring in enumerate(rings):
            if holes[i]:
                continue
            d = "M " + " L ".join(f"{a},{b}" for a, b in mm(ring[:-1])) + " Z"
            for j, hole in enumerate(rings):
                if holes[j] and parents[j] == i:
                    d += " M " + " L ".join(f"{a},{b}" for a, b in mm(hole[:-1])) + " Z"
            path_el = dwg.path(d=d)
            if ring_w is not None and i < len(ring_w):
                path_el.update({"stroke-width": round(ring_w[i] * mm_per_px, 2)})
            grp.add(path_el)
            n += 1
        # orphan holes (parent ring was dropped): draw as outline only
        for j, ring in enumerate(rings):
            if holes[j] and parents[j] == -1:
                grp.add(dwg.polyline(points=mm(ring), fill="white"))
                n += 1
        dwg.add(grp)
    if style.get("pots"):
        import math
        grp = dwg.g(id="calque-pots", fill="white", stroke="#2a6e28",
                    stroke_width=round(mm_per_px, 2))
        for cx, cy, w, h, ang in style["pots"]:
            if w >= h:
                axis_deg, rx, ry = ang, w / 2.0, h / 2.0
            else:
                axis_deg, rx, ry = ang + 90.0, h / 2.0, w / 2.0
            ex = cx + math.cos(math.radians(axis_deg))
            ey = cy + math.sin(math.radians(axis_deg))
            (cxm, cym), (exm, eym) = mm([(cx, cy), (ex, ey)])
            rot = math.degrees(math.atan2(eym - cym, exm - cxm))
            grp.add(dwg.ellipse(center=(cxm, cym),
                                r=(round(rx * mm_per_px, 2), round(ry * mm_per_px, 2)),
                                transform=f"rotate({rot:.1f} {cxm} {cym})"))
            n += 1
        dwg.add(grp)
    for name in LINE_LAYERS:
        polys = layers.get(name, [])
        widths = style.get("width", {}).get(name, [1.5] * len(polys))
        tones = style.get("tone", {}).get(name, [40.0] * len(polys))
        grp = dwg.g(id=f"calque-{name}", fill="none", stroke_linecap="round")
        for poly, wpx, t in zip(polys, widths, tones):
            g = int(np.clip(t, 0, 255))
            grp.add(dwg.polyline(points=mm(poly), stroke=f"#{g:02x}{g:02x}{g:02x}",
                                 stroke_width=round(wpx * mm_per_px, 2)))
            n += 1
        dwg.add(grp)
    dwg.save()
    return n
