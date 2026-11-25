"""Extract layered centerline vectors from a raster architecture drawing.

Strategy per tonal/color class:
  - "heavy"  (dark linework: walls, cut lines, outlines)  -> skeleton centerlines
  - "light"  (mid-gray linework: hatch, furniture, tiles) -> skeleton centerlines
  - "green"  (vegetation fills)                           -> region outlines

Skeletons come from cv2.ximgproc.thinning, then get traced into polylines and
simplified with Douglas-Peucker so DXF endpoints land on real drawing nodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .refine import RefineParams, refine_layers


@dataclass
class VectorizeParams:
    dark_max: int = 110          # gray <= this -> heavy linework
    light_max: int = 190         # dark_max < gray <= this -> light linework
    sat_min_color: int = 60      # saturation above which a pixel counts as colored
    green_hue: tuple[int, int] = (35, 90)
    min_len_px: dict = field(default_factory=lambda: {"heavy": 6.0, "light": 8.0, "green": 12.0, "accent": 12.0})
    simplify_eps_px: float = 1.6
    close_kernel: int = 3        # morphological closing before thinning
    refine: bool = True          # regularize skeleton paths (refine.py)
    refine_params: RefineParams = field(default_factory=RefineParams)
    region_close: int = 5        # closing that merges fill texture into clumps
    region_eps_px: float = 1.6   # DP eps for region outlines
    region_holes: bool = True    # trace interior holes (planting rows, gaps)
    interior_min_len_px: float = 40.0  # extra min length for linework inside green (0 = off)
    max_stroke_halfwidth: float = 3.5  # pale marks wider than 2x this are soft shading, not ink
    shade_regions: bool = False  # posterize soft shading into gray fill regions
    shade_max: int = 242         # gray above this is paper
    shade_split: int = 226       # two shade bands: darker / lighter than this
    pots: bool = True            # replace pot speckle with clean uniform symbols
    pot_area_px: tuple[float, float] = (20.0, 250.0)  # hole area accepted as a pot seed
    pot_white_min: int = 208     # gray at/above this counts as a white pot top
    canopy_close: int = 11       # merge that turns foliage into one canopy mass


def _masks(bgr: np.ndarray, p: VectorizeParams) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # v-gate: JPEG chroma fringing makes dark ink pixels saturated; ink wins
    # (dark strokes stay ink even over color fills -- outlines on green roofs)
    colored = (s >= p.sat_min_color) & (v >= 80)
    green = colored & (h >= p.green_hue[0]) & (h <= p.green_hue[1])
    accent = colored & ~green
    to_u8 = lambda m: (m.astype(np.uint8)) * 255
    # one union mask for ALL linework; strokes are classified heavy/light per
    # traced path afterwards (per-pixel classes flap along faint strokes and
    # anti-aliased halos, producing doubled + dashed lines).
    # dark ink only excludes truly-colored pixels (pot outlines etc. run
    # through fills); the pale band also drops a 1px halo around fills, whose
    # anti-aliased edge reads as mid-gray and would trace as a ghost ring
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    color_halo = cv2.dilate(to_u8(colored), ring) > 0
    heavy = (gray <= p.dark_max) & ~colored
    light = (gray > p.dark_max) & (gray <= p.light_max) & ~color_halo
    # broad pale areas are soft shading (drop shadows, gradients), not strokes:
    # opening keeps only structures wider than a real stroke; remove those
    blob = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(2 * p.max_stroke_halfwidth) + 2,) * 2)
    broad_pale = cv2.morphologyEx(to_u8(light), cv2.MORPH_OPEN, blob) > 0
    light = light & ~broad_pale
    # everything broad and non-white becomes the shade mask (posterized fills)
    shade_all = ~colored & (gray > p.dark_max) & (gray <= p.shade_max)
    shade = cv2.morphologyEx(to_u8(shade_all), cv2.MORPH_OPEN, blob) > 0
    return {"heavy": to_u8(heavy), "light": to_u8(light),
            "line": to_u8(heavy | light),
            "green": to_u8(green), "accent": to_u8(accent),
            "shade": to_u8(shade), "_gray": gray}


def _trace_skeleton(skel: np.ndarray) -> list[np.ndarray]:
    """Walk an 8-connected skeleton into pixel polylines (near-linear)."""
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pts = np.argwhere(skel > 0)          # (row, col)
    alive = set(map(tuple, pts))

    def neighbors(p):
        r, c = p
        return [(r + dr, c + dc) for dr, dc in nbrs if (r + dr, c + dc) in alive]

    # degree-1 pixels once, up front; walks start there (loops handled after)
    endpoints = [p for p in alive if len(neighbors(p)) == 1]
    from collections import deque
    queue = deque(endpoints)

    paths = []

    def walk(start):
        path = [start]
        alive.discard(start)
        cur = start
        while True:
            nxt = neighbors(cur)
            if not nxt:
                break
            cur = nxt[0]
            path.append(cur)
            alive.discard(cur)
            if len(nxt) > 1:
                # junction: other branches become fresh endpoints
                queue.extend(n for n in nxt[1:] if n in alive)
        paths.append(np.array([[c, r] for r, c in path], dtype=np.float64))

    while queue:
        p = queue.popleft()
        if p in alive:
            walk(p)
    while alive:                          # remaining pure loops
        walk(next(iter(alive)))
    return paths


def _simplify(paths: list[np.ndarray], eps: float, min_len: float) -> list[np.ndarray]:
    out = []
    for path in paths:
        if len(path) < 2:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        if length < min_len:
            continue
        approx = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2), eps, False)
        out.append(approx.reshape(-1, 2).astype(np.float64))
    return out


def _sample_along(poly: np.ndarray, img: np.ndarray, step: float = 2.0) -> float:
    """Median of img sampled along a polyline (poly in img pixel coords)."""
    h, w = img.shape[:2]
    vals = []
    for a, b in zip(poly, poly[1:]):
        seg = b - a
        n = max(1, int(np.linalg.norm(seg) / step))
        for t in np.arange(n) / n:
            x, y = (a + t * seg).astype(int)
            if 0 <= y < h and 0 <= x < w:
                vals.append(float(img[y, x]))
    return float(np.median(vals)) if vals else 0.0


def _smooth_ring(c: np.ndarray, eps: float, offset: np.ndarray) -> np.ndarray | None:
    """Contour -> closed, de-staircased, simplified ring polyline."""
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) >= 9:
        avg = np.ones(5) / 5.0
        pts = np.column_stack([
            np.convolve(np.concatenate([pts[-2:, k], pts[:, k], pts[:2, k]]),
                        avg, mode="same")[2:-2] for k in (0, 1)])
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
    poly = approx.reshape(-1, 2).astype(np.float64) + offset
    if len(poly) < 3:
        return None
    return np.vstack([poly, poly[:1]])


def _ellipse_poly(e: tuple) -> np.ndarray:
    """(cx, cy, w, h, angle_deg) -> closed 24-gon approximation."""
    cx, cy, w, h, ang = e
    t = np.linspace(0.0, 2.0 * np.pi, 25)
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    ex, ey = 0.5 * w * np.cos(t), 0.5 * h * np.sin(t)
    return np.column_stack([cx + ex * ca - ey * sa, cy + ex * sa + ey * ca])


def _green_canopy(fine: np.ndarray, p: VectorizeParams, offset: np.ndarray):
    """Clean tree treatment: smooth canopy outlines from a merged mask, pot
    symbols fitted as ellipses to the white holes of the fine mask, and
    larger interior gaps kept as smooth structure rings."""
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.canopy_close,) * 2)
    coarse = cv2.morphologyEx(fine, cv2.MORPH_CLOSE, ck)
    coarse = cv2.threshold(cv2.GaussianBlur(coarse, (5, 5), 0), 127, 255,
                           cv2.THRESH_BINARY)[1]
    rings, holes, parents = [], [], []
    outer_contours = [c for c in cv2.findContours(
        coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if cv2.arcLength(c, True) >= p.min_len_px.get("green", 12.0)]
    for c in outer_contours:
        poly = _smooth_ring(c, p.region_eps_px, offset)
        if poly is not None:
            rings.append(poly)
            holes.append(False)
            parents.append(-1)
    n_outer = len(rings)

    pots = []
    contours, hierarchy = cv2.findContours(fine, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0] if hierarchy is not None else []
    for i, c in enumerate(contours):
        if len(hierarchy) == 0 or hierarchy[i][3] == -1:
            continue                                   # holes only
        area = cv2.contourArea(c)
        if p.pot_area_px[0] <= area <= p.pot_area_px[1] and len(c) >= 5:
            (cx, cy), (w, h), ang = cv2.fitEllipse(c)
            if min(w, h) > 0 and max(w, h) / min(w, h) <= 2.2:
                pots.append((cx + offset[0], cy + offset[1], w, h, ang))
                continue
        if cv2.arcLength(c, True) >= 60.0:             # larger gap: keep shape
            poly = _smooth_ring(c, p.region_eps_px, offset)
            if poly is not None:
                parent = -1
                pt = tuple(float(v) for v in c[0][0])
                for j in range(n_outer):
                    if cv2.pointPolygonTest(outer_contours[j], pt, False) >= 0:
                        parent = j
                        break
                rings.append(poly)
                holes.append(True)
                parents.append(parent)

    filled = np.zeros_like(coarse)
    cv2.drawContours(filled, outer_contours, -1, 255, cv2.FILLED)
    return rings, holes, parents, pots, filled


def _pot_instances(gray: np.ndarray, seeds: list[tuple], region: np.ndarray,
                   offset: np.ndarray, white_min: int = 222) -> list[tuple]:
    """One drawn symbol, many instances: pot tops are the only WHITE blobs
    inside the canopy, each separated by its dark rim.  Distance-transform
    peaks of the white mask give one center per pot even when foliage hides
    most of it; every center gets the same median-sized ellipse symbol."""
    if len(seeds) < 8:
        return seeds
    w_m = float(np.median([s[2] for s in seeds]))
    h_m = float(np.median([s[3] for s in seeds]))
    ang_m = float(np.median([s[4] if s[4] <= 90 else s[4] - 180 for s in seeds]))
    white = (((gray >= white_min) & (region > 0)).astype(np.uint8)) * 255
    # a pot top is a small isolated white blob; big pools are paper/plaza gaps
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(white)
    areas = stats[:, cv2.CC_STAT_AREA]
    bad = (areas < 6) | (areas > 500)
    bad[0] = True
    white[bad[lbl]] = 0
    dt = cv2.distanceTransform(white, cv2.DIST_L2, 3)
    rmax = max(w_m, h_m) / 2.0 + 2.0        # bigger white pools are paper gaps
    local_max = dt >= cv2.dilate(dt, np.ones((5, 5), np.uint8)) - 1e-6
    cand = local_max & (dt >= 1.3) & (dt <= rmax)
    ys, xs = np.nonzero(cand)
    order = np.argsort(-dt[ys, xs])
    min_d2 = (0.62 * max(w_m, h_m)) ** 2
    pots: list[tuple] = []
    placed: list[tuple[float, float]] = []
    for k in order:
        x, y = float(xs[k]), float(ys[k])
        if any((x - px) ** 2 + (y - py) ** 2 < min_d2 for px, py in placed):
            continue
        placed.append((x, y))
        pots.append((x + offset[0], y + offset[1], w_m, h_m, ang_m))
    return pots


def _split_by_width(poly: np.ndarray, halfwidth: np.ndarray,
                    min_piece_px: float = 30.0) -> list[np.ndarray]:
    """Split a polyline where the measured stroke weight changes class, so a
    chain that runs bold-then-thin doesn't get one averaged width.  Runs
    shorter than min_piece_px merge into a neighbor -- weight flicker along a
    stroke must not shatter it into stubs."""
    if len(poly) < 3:
        return [poly]
    seg_len = [float(np.linalg.norm(b - a)) for a, b in zip(poly, poly[1:])]
    widths = [2.0 * _sample_along(np.array([a, b]), halfwidth)
              for a, b in zip(poly, poly[1:])]
    cls = [0 if w < 1.9 else (2 if w > 3.0 else 1) for w in widths]
    sm = [sorted(cls[max(0, i - 1):i + 2])[len(cls[max(0, i - 1):i + 2]) // 2]
          for i in range(len(cls))]         # 3-tap median: kill flicker
    # runs of equal class over segment indices
    runs = []                               # [cls, seg_start, seg_end, length]
    for i, c in enumerate(sm):
        if runs and runs[-1][0] == c:
            runs[-1][2] = i + 1
            runs[-1][3] += seg_len[i]
        else:
            runs.append([c, i, i + 1, seg_len[i]])
    # absorb short runs into the longer adjacent run until all are substantial
    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda i: runs[i][3])
        if runs[k][3] >= min_piece_px:
            break
        left = runs[k - 1][3] if k > 0 else -1.0
        right = runs[k + 1][3] if k + 1 < len(runs) else -1.0
        j = k - 1 if left >= right else k + 1
        a, b = (j, k) if j < k else (k, j)
        runs[a] = [runs[a][0] if runs[a][3] >= runs[b][3] else runs[b][0],
                   runs[a][1], runs[b][2], runs[a][3] + runs[b][3]]
        del runs[b]
    return [poly[r[1]:r[2] + 1] for r in runs if len(poly[r[1]:r[2] + 1]) >= 2]


def extract(image_path: str, params: VectorizeParams | None = None,
            crop: tuple[int, int, int, int] | None = None,
            return_style: bool = False):
    """Return {layer: [polyline Nx2 arrays in pixel coords], ...}.

    crop: optional (x0, y0, x1, y1) region of interest; output stays in
    full-image pixel coordinates so one calibration serves any crop.
    return_style: also return a style dict -- per-polyline stroke width (px)
    and tone (gray) for line layers, hole flags + parent index and fill color
    for region layers -- so exports can look like the source drawing.
    """
    p = params or VectorizeParams()
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    offset = np.array([0.0, 0.0])
    if crop:
        x0, y0, x1, y1 = crop
        bgr = bgr[y0:y1, x0:x1]
        offset = np.array([float(x0), float(y0)])

    masks = _masks(bgr, p)
    gray = masks.pop("_gray")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_kernel, p.close_kernel))
    layers: dict[str, list[np.ndarray]] = {}
    raw_lines: dict[str, list[np.ndarray]] = {}
    style: dict = {"width": {}, "tone": {}, "hole": {}, "parent": {}, "fill": {}}

    # color-fill regions first: their footprints also gate the linework pass
    region_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (p.region_close, p.region_close))
    region_fill = {}
    region_names = ["green", "accent"]
    shade = masks.pop("shade")
    if p.shade_regions:
        masks["shade_dark"] = ((shade > 0) & (gray <= p.shade_split)).astype(np.uint8) * 255
        masks["shade_light"] = ((shade > 0) & (gray > p.shade_split)).astype(np.uint8) * 255
        region_names += ["shade_dark", "shade_light"]
    for name in region_names:
        m = cv2.morphologyEx(masks[name], cv2.MORPH_OPEN, kernel)     # despeckle
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, region_kernel)       # merge texture
        m = cv2.threshold(cv2.GaussianBlur(m, (3, 3), 0), 127, 255,
                          cv2.THRESH_BINARY)[1]                       # round jaggies
        src_px = bgr[masks[name] > 0]
        fill_color = tuple(int(x) for x in np.median(src_px, axis=0)) \
            if len(src_px) else (255, 255, 255)
        if name == "green" and p.pots:
            rings, holes, parents, pots, filled = _green_canopy(m, p, offset)
            layers["green"] = rings
            style["hole"]["green"] = holes
            style["parent"]["green"] = parents
            style["fill"]["green"] = fill_color
            style["width"]["green"] = [1.0 if h else 2.5 for h in holes]
            pots = _pot_instances(gray, pots, cv2.dilate(filled, region_kernel),
                                  offset, p.pot_white_min)
            if pots:
                style["pots"] = pots
                layers["pots"] = [_ellipse_poly(e) for e in pots]
            region_fill["green"] = cv2.dilate(filled, region_kernel)
            continue
        mode = cv2.RETR_CCOMP if p.region_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(m, mode, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0] if hierarchy is not None else []
        rings = []
        outer = []
        holes = []
        parents = []
        ring_idx_of_contour: dict[int, int] = {}
        for i, c in enumerate(contours):
            if cv2.arcLength(c, True) < p.min_len_px.get(name, 40.0):
                continue
            is_hole = len(hierarchy) > 0 and hierarchy[i][3] != -1
            if not is_hole:
                outer.append(c)
            pts = c.reshape(-1, 2).astype(np.float64)
            if len(pts) >= 9:   # circular moving average: de-staircase the ring
                avg = np.ones(5) / 5.0
                pts = np.column_stack([
                    np.convolve(np.concatenate([pts[-2:, k], pts[:, k], pts[:2, k]]),
                                avg, mode="same")[2:-2] for k in (0, 1)])
            approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                                      p.region_eps_px, True)
            poly = approx.reshape(-1, 2).astype(np.float64) + offset
            if len(poly) >= 3:
                ring_idx_of_contour[i] = len(rings)
                rings.append(np.vstack([poly, poly[:1]]))   # close the ring
                holes.append(is_hole)
                parents.append(hierarchy[i][3] if is_hole else -1)
        # remap hole parents from contour index to kept-ring index
        parents = [ring_idx_of_contour.get(pi, -1) for pi in parents]
        layers[name] = rings
        style["hole"][name] = holes
        style["parent"][name] = parents
        style["fill"][name] = fill_color
        fill = np.zeros_like(m)
        cv2.drawContours(fill, outer, -1, 255, cv2.FILLED)  # solid, holes included
        # dilate so the region's own outline stroke counts as "inside"
        region_fill[name] = cv2.dilate(fill, region_kernel)

    # crater rims: the roof-cut edge around a pot terrace is drawn under the
    # foliage, so its stroke arrives shredded; rebuild it as one smooth closed
    # curve from the crater region boundary and drop the shreds along it
    rim_rings: list[np.ndarray] = []
    rim_band = None
    if p.pots and style.get("pots") and region_fill.get("green") is not None:
        crater_src = region_fill["green"].copy()
        if region_fill.get("accent") is not None:
            crater_src = crater_src | region_fill["accent"]
        crater = cv2.morphologyEx(crater_src, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        pots_xy = [(float(e[0] - offset[0]), float(e[1] - offset[1]))
                   for e in style["pots"]]
        contours, _ = cv2.findContours(crater, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        band = np.zeros_like(crater)
        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue
            inside = sum(1 for x, y in pots_xy
                         if cv2.pointPolygonTest(c, (x, y), False) >= 0)
            if inside < 3:
                continue                    # a grove, not a pot terrace
            ring = _smooth_ring(c, 2.0, offset)
            if ring is None:
                continue
            rim_rings.append(ring)
            cv2.polylines(band, [c.reshape(-1, 1, 2)], True, 255, 11)
        rim_band = band if rim_rings else None

    skeleton_classes = []
    for name in ("heavy", "light"):
        m = cv2.morphologyEx(masks[name], cv2.MORPH_CLOSE, kernel)
        coverage = float(np.count_nonzero(m)) / m.size
        if coverage > 0.12:
            # broad filled/shaded areas (renders, diagrams): outline regions
            # instead of skeletonizing a massive blob
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rings = []
            for c in contours:
                if cv2.arcLength(c, True) < p.min_len_px[name]:
                    continue
                approx = cv2.approxPolyDP(c, p.simplify_eps_px, True)
                poly = approx.reshape(-1, 2).astype(np.float64) + offset
                if len(poly) >= 3:
                    rings.append(np.vstack([poly, poly[:1]]))
            layers[name] = rings
            halfwidth = cv2.dilate(cv2.distanceTransform(m, cv2.DIST_L2, 3),
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            max_w = 3.0 if name == "light" else 8.0
            style["width"][name] = [
                float(np.clip(2.0 * _sample_along(ring - offset, halfwidth), 1.0, max_w))
                for ring in rings]
            style["tone"][name] = [_sample_along(ring - offset, gray) for ring in rings]
            continue
        skeleton_classes.append(name)

    if skeleton_classes:
        # skeletonize ONE union mask, then classify each stroke by its median
        # gray -- per-stroke, so faint lines can't flap between classes
        union = masks["line"] if len(skeleton_classes) == 2 else masks[skeleton_classes[0]]
        m = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
        skel = cv2.ximgproc.thinning(m)
        # max-filtered so samples slightly off the mask still see stroke width
        halfwidth = cv2.dilate(cv2.distanceTransform(m, cv2.DIST_L2, 3),
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        raw_lines = {name: [] for name in skeleton_classes}
        green_fill = region_fill.get("green")
        for path in _trace_skeleton(skel):
            if len(path) < 2:
                continue
            xs = path[:, 0].astype(int)
            ys = path[:, 1].astype(int)
            tone = float(np.median(gray[ys, xs]))
            name = "heavy" if (tone <= p.dark_max and "heavy" in raw_lines) \
                else ("light" if "light" in raw_lines else "heavy")
            length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
            min_len = p.min_len_px[name]
            if rim_band is not None and \
                    float(np.mean(rim_band[ys, xs] > 0)) > 0.5:
                continue    # shredded rim stroke; the rebuilt rim owns this edge
            if green_fill is not None and \
                    float(np.mean(green_fill[ys, xs] > 0)) > 0.35:
                if float(np.median(halfwidth[ys, xs])) > p.max_stroke_halfwidth:
                    continue    # terrace shadow blob inside canopy, not a line
                if p.interior_min_len_px > 0:
                    min_len = max(min_len, p.interior_min_len_px)  # foliage texture
            if length >= min_len:
                raw_lines[name].append(path)

    if raw_lines:
        if p.refine:
            # one refine call across layers so dominant angles are shared
            refined = refine_layers(raw_lines, p.refine_params)
        else:
            refined = {name: _simplify(paths, p.simplify_eps_px, p.min_len_px[name])
                       for name, paths in raw_lines.items()}
        for name, polys in refined.items():
            layers[name] = []
            style["width"][name] = []
            style["tone"][name] = []
            max_w = 3.0 if name == "light" else 8.0   # pale lines are thin by nature
            for pl in polys:
                for piece in _split_by_width(pl, halfwidth):
                    layers[name].append(piece + offset)
                    style["width"][name].append(
                        float(np.clip(2.0 * _sample_along(piece, halfwidth), 1.0, max_w)))
                    style["tone"][name].append(_sample_along(piece, gray))

    # a pale line hugging a dark one for its whole length is the dark line's
    # anti-aliased/shadow rim, not a second drawn line -- drop it
    if layers.get("heavy") and layers.get("light"):
        hv = np.zeros(gray.shape, np.uint8)
        for pl in layers["heavy"]:
            cv2.polylines(hv, [(pl - offset).astype(np.int32).reshape(-1, 1, 2)],
                          False, 255, 1)
        dist = cv2.distanceTransform(255 - hv, cv2.DIST_L2, 3)
        keep = [i for i, pl in enumerate(layers["light"])
                if _sample_along(pl - offset, dist) >= 2.5]
        layers["light"] = [layers["light"][i] for i in keep]
        style["width"]["light"] = [style["width"]["light"][i] for i in keep]
        style["tone"]["light"] = [style["tone"]["light"][i] for i in keep]

    for ring in rim_rings:      # rebuilt crater rims: one bold continuous curve
        layers.setdefault("heavy", []).append(ring)
        style["width"].setdefault("heavy", []).append(3.2)
        style["tone"].setdefault("heavy", []).append(30.0)

    return (layers, style) if return_style else layers
