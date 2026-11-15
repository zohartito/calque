"""Raster->vector fidelity eval: does the trace LOOK like the drawing?

Metrics (pixel space, per drawing):
  - visual_mae:     mean |source - styled render| over marked pixels, after a
                    small blur -- the "would a human say these match" gate.
                    The styled render uses measured stroke widths/tones and
                    region fills, so weight and fills count, not just position
  - line_recall:    % of ink-skeleton pixels within TOL px of a drawn vector
  - line_precision: % of vector pixels within TOL px of any source ink
  - green_iou:      filled green rings vs solidified source green mask
  - debug images:   *_debug.jpg miss overlay (magenta = lost ink),
                    *_styled.png the styled render, *_visdiff.jpg heatmap

Usage:
  python evals/eval_fidelity.py IMAGE --light-max N
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calque.vectorize import VectorizeParams, extract  # noqa: E402
from calque.export import render_styled  # noqa: E402

TOL = 2.5          # px: a stroke is "kept" if a vector runs within this
PASS_RECALL = 0.90
PASS_PRECISION = 0.90
PASS_GREEN_IOU = 0.80
PASS_VISUAL_MAE = 24.0   # 0-255 tonal units over marked pixels


def ink_masks(bgr: np.ndarray, p: VectorizeParams):
    """Source ink (any drawn line pixel) and green fill, straight from pixels."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    colored = (s >= p.sat_min_color) & (v >= 80)
    green = colored & (h >= p.green_hue[0]) & (h <= p.green_hue[1])
    ink = (gray <= p.light_max) & ~colored
    any_mark = (gray <= p.light_max) | colored
    to_u8 = lambda m: m.astype(np.uint8) * 255
    return to_u8(ink), to_u8(green), to_u8(any_mark)


def rasterize(layers: dict, shape, names=None) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    for name, polys in layers.items():
        if names is not None and name not in names:
            continue
        for poly in polys:
            cv2.polylines(canvas, [poly.astype(np.int32).reshape(-1, 1, 2)],
                          False, 255, 1)
    return canvas


def run(image_path: str, params: VectorizeParams, debug_out: str | None):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    layers, style = extract(image_path, params, return_style=True)
    ink, green_src, any_mark = ink_masks(bgr, params)

    # ---- perceptual: styled render vs source ----
    styled = render_styled(layers, style, bgr.shape[:2])
    src_b = cv2.GaussianBlur(bgr, (5, 5), 1.5).astype(np.float32)
    ren_b = cv2.GaussianBlur(styled, (5, 5), 1.5).astype(np.float32)
    marked = (src_b.min(axis=2) < 235) | (ren_b.min(axis=2) < 235)
    diff = np.abs(src_b - ren_b).mean(axis=2)
    visual_mae = float(diff[marked].mean()) if marked.any() else 0.0
    visual_p95 = float(np.percentile(diff[marked], 95)) if marked.any() else 0.0

    # shade layers are fills judged by visual_mae, not linework claims
    vec = rasterize(layers, ink.shape,
                    names=("heavy", "light", "green", "accent", "pots"))
    # recall: ink skeleton pixels near any vector. Width-gate the reference:
    # pale marks wider than a stroke are soft shading, not drawn lines
    ink_skel = cv2.ximgproc.thinning(ink)
    halfwidth = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    d_vec = cv2.distanceTransform(255 - vec, cv2.DIST_L2, 3)
    skel_pts = (ink_skel > 0) & ((halfwidth <= params.max_stroke_halfwidth)
                                 | (gray <= params.dark_max))
    # inside vegetation, small ink specks are foliage/pot texture that the
    # deliverable renders as symbols -- only sustained linework is required
    k9 = np.ones((9, 9), np.uint8)
    green_zone = cv2.dilate(green_src, k9) > 0
    n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8))
    drop = np.zeros(n_lbl, bool)
    for i in range(1, n_lbl):
        cx, cy = int(cents[i][0]), int(cents[i][1])
        if stats[i, cv2.CC_STAT_AREA] < 120 and green_zone[cy, cx]:
            drop[i] = True
    skel_pts &= ~drop[lbl]
    recall = float((d_vec[skel_pts] <= TOL).mean()) if skel_pts.any() else 1.0

    # precision: vector pixels near any source mark (ink or fill)
    d_ink = cv2.distanceTransform(255 - any_mark, cv2.DIST_L2, 3)
    vec_pts = vec > 0
    precision = float((d_ink[vec_pts] <= TOL).mean()) if vec_pts.any() else 1.0

    # green shape: filled rings vs solidified source green
    green_fill = np.zeros_like(green_src)
    for poly in layers.get("green", []):
        cv2.fillPoly(green_fill, [poly.astype(np.int32).reshape(-1, 1, 2)], 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_ref = cv2.morphologyEx(green_src, cv2.MORPH_CLOSE, k)
    inter = np.count_nonzero((green_fill > 0) & (green_ref > 0))
    union = np.count_nonzero((green_fill > 0) | (green_ref > 0))
    green_iou = inter / union if union else 1.0

    if debug_out:
        Path(debug_out).parent.mkdir(parents=True, exist_ok=True)
        over = cv2.addWeighted(bgr, 0.25, np.full_like(bgr, 255), 0.75, 0)
        missed = skel_pts & (d_vec > TOL)
        spurious = vec_pts & (d_ink > TOL)
        over[missed] = (200, 0, 200)     # magenta: lost ink
        over[spurious] = (200, 200, 0)   # cyan: hallucinated vectors
        near = vec_pts & (d_ink <= TOL)
        over[near] = (60, 60, 60)
        cv2.imwrite(debug_out, over)
        stem = str(Path(debug_out).with_suffix(""))
        cv2.imwrite(stem + "_styled.png", styled)
        heat = np.clip(diff * 3, 0, 255).astype(np.uint8)
        cv2.imwrite(stem + "_visdiff.jpg",
                    cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO))

    return {
        "image": Path(image_path).name,
        "visual_mae": round(visual_mae, 2),
        "visual_p95": round(visual_p95, 2),
        "line_recall": round(recall, 4),
        "line_precision": round(precision, 4),
        "green_iou": round(green_iou, 4),
        "n_polylines": {k: len(v) for k, v in layers.items()},
        "pass": bool(visual_mae <= PASS_VISUAL_MAE
                     and recall >= PASS_RECALL and precision >= PASS_PRECISION
                     and green_iou >= PASS_GREEN_IOU),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="input raster drawing (required)")
    ap.add_argument("--light-max", type=int, default=205)
    ap.add_argument("--debug", help="optional debug image path; creates parent directories")
    args = ap.parse_args(argv)
    # shade on: the visual gate compares the fully-styled render to the source
    params = VectorizeParams(light_max=args.light_max, shade_regions=True)
    try:
        result = run(args.image, params, args.debug)
    except (OSError, ValueError, cv2.error) as exc:
        ap.error(f"cannot evaluate drawing: {exc}")
    print(json.dumps(result, indent=2))
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
