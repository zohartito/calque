"""Regularize traced skeleton paths into clean, CAD-editable linework.

Pixel-space post-processing between skeleton tracing and export:
  1. split raw pixel paths at corners; fit each run with total least squares
  2. snap near-dominant run directions to the drawing's global angle peaks
     (long/isolated runs only, so curves keep their shape)
  3. rebuild corner vertices as line-line intersections (crisp corners)
  4. bridge collinear gaps between free segments (broken strokes -> one line)
  5. snap endpoint clusters and T-junctions so osnap lands on shared nodes
  6. chain polylines that meet end-to-end into single entities
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RefineParams:
    dp_eps_px: float = 1.3        # DP eps for candidate breakpoints
    fit_tol_px: float = 1.4       # max TLS residual for one straight run
    corner_deg: float = 28.0      # turn sharper than this always breaks a run
    snap_angle_deg: float = 4.0   # snap runs within this of a dominant angle
    min_snap_len_px: float = 8.0  # never snap runs shorter than this
    curve_snap_len_px: float = 45.0  # curve-end runs shorter than this stay curve
    merge_gap_px: float = 8.0     # bridge collinear endpoint gaps up to this
    lateral_tol_px: float = 1.75  # max sideways offset for a collinear merge
    join_radius_px: float = 5.0   # endpoint clusters within this become one node
    t_snap_px: float = 4.0        # endpoint-to-edge distance for T-junction snap


# ---------------------------------------------------------------- fitting

def _fit_line(pts: np.ndarray):
    """TLS fit -> (centroid, unit direction, max residual)."""
    c = pts.mean(axis=0)
    q = pts - c
    if len(pts) == 2:
        d = q[1] - q[0]
        n = float(np.linalg.norm(d))
        d = d / n if n else np.array([1.0, 0.0])
        return c, d, 0.0
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    res = np.abs(q @ np.array([-d[1], d[0]]))
    return c, d, float(res.max())


def _turn_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cosv))


def _dir_diff_deg(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle between undirected directions (mod 180)."""
    cosv = abs(float(np.clip(np.dot(d1, d2), -1.0, 1.0)))
    return math.degrees(math.acos(cosv))


def _intersect(c1, d1, c2, d2):
    """Intersection of two infinite lines, or None if near-parallel."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


# ---------------------------------------------------------------- runs

@dataclass
class _Run:
    c: np.ndarray        # centroid of fitted line
    d: np.ndarray        # unit direction
    e0: np.ndarray       # fitted start point
    e1: np.ndarray       # fitted end point
    length: float
    i0: int = 0          # raw-path index range this run covers
    i1: int = 0
    gradual_prev: bool = False   # joins previous run with a gentle turn (curve)
    gradual_next: bool = False

    def curvey(self, p: "RefineParams") -> bool:
        if self.gradual_prev and self.gradual_next:
            return True          # mid-curve chord, whatever its length
        return (self.gradual_prev or self.gradual_next) and \
            self.length < p.curve_snap_len_px


def _breakpoints(path: np.ndarray, eps: float) -> list[int]:
    approx = cv2.approxPolyDP(path.astype(np.float32).reshape(-1, 1, 2),
                              eps, False).reshape(-1, 2)
    idx, j = [], 0
    for a in approx:
        while j < len(path) and not (path[j, 0] == a[0] and path[j, 1] == a[1]):
            j += 1
        if j == len(path):
            break
        idx.append(j)
        j += 1
    if not idx or idx[0] != 0:
        idx.insert(0, 0)
    if idx[-1] != len(path) - 1:
        idx.append(len(path) - 1)
    return idx


def _split_runs(path: np.ndarray, p: RefineParams) -> list[_Run]:
    """Greedy corner/fit-guided split of one raw pixel path into line runs."""
    bs = _breakpoints(path, p.dp_eps_px)
    spans: list[tuple[int, int]] = []
    s, e = 0, 1
    while e < len(bs):
        nxt = e + 1
        extended = False
        if nxt < len(bs):
            turn = _turn_deg(path[bs[e]] - path[bs[e - 1]],
                             path[bs[nxt]] - path[bs[e]])
            if turn < p.corner_deg:
                _, _, res = _fit_line(path[bs[s]:bs[nxt] + 1])
                if res <= p.fit_tol_px:
                    e = nxt
                    extended = True
        if not extended:
            spans.append((bs[s], bs[e]))
            s, e = e, e + 1

    runs: list[_Run] = []
    for i0, i1 in spans:
        pts = path[i0:i1 + 1]
        c, d, _ = _fit_line(pts)
        t0, t1 = float((pts[0] - c) @ d), float((pts[-1] - c) @ d)
        e0, e1 = c + t0 * d, c + t1 * d
        runs.append(_Run(c, d, e0, e1, float(abs(t1 - t0)), i0, i1))
    for k in range(1, len(runs)):
        turn = _turn_deg(runs[k - 1].e1 - runs[k - 1].e0,
                         runs[k].e1 - runs[k].e0)
        if turn < p.corner_deg:
            runs[k - 1].gradual_next = True
            runs[k].gradual_prev = True
    return runs


# ---------------------------------------------------------------- angles

def _dominant_angles(all_runs: list[_Run], min_share: float = 0.02) -> list[float]:
    """Length-weighted circular histogram (deg mod 180) -> peak angles."""
    hist = np.zeros(180)
    total = 0.0
    for r in all_runs:
        ang = math.degrees(math.atan2(r.d[1], r.d[0])) % 180.0
        hist[int(ang) % 180] += r.length
        total += r.length
    if total <= 0:
        return []
    kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
    sm = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), kernel,
                     mode="same")[2:-2]
    peaks = []
    for b in range(180):
        window = sm[[(b + o) % 180 for o in range(-3, 4)]]
        if sm[b] >= window.max() and sm[b] > total * min_share:
            # sub-bin refinement: weighted mean over the window
            offs = np.arange(-3, 4)
            w = window
            peaks.append((b + float((offs * w).sum() / w.sum())) % 180.0)
    # collapse peaks closer than 3 deg (keep the stronger)
    peaks.sort(key=lambda a: -sm[int(a) % 180])
    kept: list[float] = []
    for a in peaks:
        if all(min(abs(a - k), 180 - abs(a - k)) > 3.0 for k in kept):
            kept.append(a)
    return kept


def _snap_runs(runs_by_path: list[list[_Run]], p: RefineParams) -> None:
    flat = [r for runs in runs_by_path for r in runs]
    peaks = _dominant_angles(flat)
    if not peaks:
        return
    for runs in runs_by_path:
        for r in runs:
            if r.length < p.min_snap_len_px or r.curvey(p):
                continue
            ang = math.degrees(math.atan2(r.d[1], r.d[0])) % 180.0
            best = min(peaks, key=lambda a: min(abs(ang - a), 180 - abs(ang - a)))
            # cap so snapping never moves an endpoint more than ~2.5 px
            eff_tol = min(p.snap_angle_deg,
                          math.degrees(math.atan2(5.0, r.length)))
            if min(abs(ang - best), 180 - abs(ang - best)) > eff_tol:
                continue
            nd = np.array([math.cos(math.radians(best)),
                           math.sin(math.radians(best))])
            if np.dot(nd, r.d) < 0:
                nd = -nd
            r.d = nd
            t0, t1 = float((r.e0 - r.c) @ nd), float((r.e1 - r.c) @ nd)
            r.e0, r.e1 = r.c + t0 * nd, r.c + t1 * nd
            r.length = abs(t1 - t0)


def _smooth_open(pts: np.ndarray, window: int = 5, dp_eps: float = 0.9) -> np.ndarray:
    """Moving-average smooth + light DP; keeps curves round, kills jitter."""
    if len(pts) <= window:
        return pts
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    sm = np.column_stack([np.convolve(padded[:, k], kernel, mode="valid")
                          for k in (0, 1)])
    sm[0], sm[-1] = pts[0], pts[-1]
    approx = cv2.approxPolyDP(sm.astype(np.float32).reshape(-1, 1, 2),
                              dp_eps, False)
    return approx.reshape(-1, 2).astype(np.float64)


def _path_geometry(path: np.ndarray, runs: list[_Run], p: RefineParams) -> np.ndarray:
    """One polyline for a multi-run path: straight runs become fitted edges
    with intersection corners; chains of short gradual runs (curves) keep
    their smoothed raw shape instead of chord facets."""
    verts: list[np.ndarray] = []
    i = 0
    while i < len(runs):
        if runs[i].curvey(p):
            j = i
            while j + 1 < len(runs) and runs[j].gradual_next and runs[j + 1].curvey(p):
                j += 1
            pts = _smooth_open(path[runs[i].i0:runs[j].i1 + 1])
            start = 1 if verts else 0        # previous run already placed a joint
            verts.extend(pts[start:])
            i = j + 1
            continue
        r = runs[i]
        if not verts:
            verts.append(r.e0)
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if nxt is not None and not nxt.curvey(p):
            joint = 0.5 * (r.e1 + nxt.e0)
            if _dir_diff_deg(r.d, nxt.d) >= 12.0:
                x = _intersect(r.c, r.d, nxt.c, nxt.d)
                if x is not None and np.linalg.norm(x - joint) <= 8.0:
                    joint = x
            verts.append(joint)
        else:
            verts.append(r.e1)
        i += 1
    return np.array(verts)


# ---------------------------------------------------------------- merging

def _grid(points: np.ndarray, cell: float) -> dict[tuple[int, int], list[int]]:
    g: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(points):
        g[(int(x // cell), int(y // cell))].append(i)
    return g


def _near(g, points, x, y, cell, radius) -> list[int]:
    cx, cy = int(x // cell), int(y // cell)
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for i in g.get((cx + dx, cy + dy), ()):
                if (points[i, 0] - x) ** 2 + (points[i, 1] - y) ** 2 <= radius ** 2:
                    out.append(i)
    return out


def _merge_free_segments(segs: list[np.ndarray], p: RefineParams) -> list[np.ndarray]:
    """Union collinear, nearly-touching 2-point segments into single lines."""
    for _ in range(3):
        n = len(segs)
        if n < 2:
            return segs
        ends = np.array([[s[0], s[-1]] for s in segs])          # n x 2 x 2
        dirs, lens = [], []
        for s in segs:
            v = s[-1] - s[0]
            L = float(np.linalg.norm(v))
            dirs.append(v / L if L else np.array([1.0, 0.0]))
            lens.append(L)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        radius = max(p.merge_gap_px, 12.0)
        flat = ends.reshape(-1, 2)
        g = _grid(flat, radius)
        merged_any = False
        for i in range(n):
            for endpt in ends[i]:
                for j2 in _near(g, flat, endpt[0], endpt[1], radius, radius):
                    j = j2 // 2
                    if j <= i:
                        continue
                    if _dir_diff_deg(dirs[i], dirs[j]) > 3.0:
                        continue
                    # combined axis, weighted by length
                    d = dirs[i] * lens[i] + dirs[j] * lens[j] * \
                        (1 if np.dot(dirs[i], dirs[j]) >= 0 else -1)
                    L = np.linalg.norm(d)
                    if L == 0:
                        continue
                    d = d / L
                    nrm = np.array([-d[1], d[0]])
                    c = 0.5 * (ends[i].mean(axis=0) + ends[j].mean(axis=0))
                    oi = float(((ends[i] - c) @ nrm).mean())
                    oj = float(((ends[j] - c) @ nrm).mean())
                    if abs(oi - oj) > p.lateral_tol_px:
                        continue
                    ti = (ends[i] - c) @ d
                    tj = (ends[j] - c) @ d
                    gap = max(tj.min() - ti.max(), ti.min() - tj.max())
                    if gap > p.merge_gap_px:
                        continue
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
                        merged_any = True
        if not merged_any:
            return segs
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        out = []
        for members in groups.values():
            if len(members) == 1:
                out.append(segs[members[0]])
                continue
            pts = np.vstack([ends[m] for m in members])
            c, d, _ = _fit_line(pts)
            t = (pts - c) @ d
            out.append(np.array([c + t.min() * d, c + t.max() * d]))
        segs = out
    return segs


# ---------------------------------------------------------------- junctions

def _join_endpoints(polys: list[np.ndarray], p: RefineParams) -> None:
    """Snap endpoint clusters to one node; sharp pairs meet at intersection."""
    eps_info = []   # (poly_idx, which_end, point, terminal dir into the poly)
    for pi, poly in enumerate(polys):
        if len(poly) < 2:
            continue
        d0 = poly[1] - poly[0]
        d1 = poly[-2] - poly[-1]
        for which, pt, d in ((0, poly[0], d0), (1, poly[-1], d1)):
            L = np.linalg.norm(d)
            eps_info.append((pi, which, pt.copy(), d / L if L else np.array([1.0, 0.0])))
    if not eps_info:
        return
    pts = np.array([e[2] for e in eps_info])
    g = _grid(pts, p.join_radius_px)
    seen = set()
    clustered = set()
    for i in range(len(eps_info)):
        if i in seen:
            continue
        idxs = _near(g, pts, pts[i, 0], pts[i, 1], p.join_radius_px,
                     p.join_radius_px)
        idxs = [j for j in idxs if j not in seen]
        if len(idxs) < 2:
            continue
        seen.update(idxs)
        clustered.update(idxs)
        target = pts[idxs].mean(axis=0)
        if len(idxs) == 2:
            a, b = (eps_info[j] for j in idxs)
            if _dir_diff_deg(a[3], b[3]) >= 20.0:
                x = _intersect(pts[idxs[0]], a[3], pts[idxs[1]], b[3])
                if x is not None and np.linalg.norm(x - target) <= 2 * p.join_radius_px:
                    target = x
        for j in idxs:
            pi, which, _, _ = eps_info[j]
            polys[pi][0 if which == 0 else -1] = target

    # T-junctions: project remaining endpoints onto a nearby foreign edge
    edges, owners = [], []
    for pi, poly in enumerate(polys):
        for a, b in zip(poly, poly[1:]):
            edges.append((a, b))
            owners.append(pi)
    if not edges:
        return
    A = np.array([e[0] for e in edges])
    B = np.array([e[1] for e in edges])
    AB = B - A
    ab2 = (AB ** 2).sum(axis=1)
    ab2[ab2 == 0] = 1e-12
    owners_arr = np.array(owners)
    for i, (pi, which, _, _) in enumerate(eps_info):
        if i in clustered:
            continue
        pt = polys[pi][0 if which == 0 else -1]
        t = ((pt - A) * AB).sum(axis=1) / ab2
        t = np.clip(t, 0.0, 1.0)
        proj = A + t[:, None] * AB
        dist2 = ((proj - pt) ** 2).sum(axis=1)
        dist2[owners_arr == pi] = np.inf
        j = int(np.argmin(dist2))
        if dist2[j] <= p.t_snap_px ** 2:
            polys[pi][0 if which == 0 else -1] = proj[j]


def _chain(polys: list[np.ndarray]) -> list[np.ndarray]:
    """Join polylines that share an endpoint nobody else uses (degree 2)."""
    key = lambda pt: (round(float(pt[0]), 2), round(float(pt[1]), 2))
    node_uses: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    for pi, poly in enumerate(polys):
        node_uses[key(poly[0])].append((pi, 0))
        node_uses[key(poly[-1])].append((pi, 1))

    parent = {}
    merged = {i: poly.copy() for i, poly in enumerate(polys)}

    def find(i):
        while parent.get(i, i) != i:
            parent[i] = parent.get(parent[i], parent[i])
            i = parent[i]
        return i

    for uses in node_uses.values():
        if len(uses) != 2:
            continue
        (p1, e1), (p2, e2) = uses
        r1, r2 = find(p1), find(p2)
        if r1 == r2:
            continue
        a, b = merged[r1], merged[r2]
        ka, kb = key(a[0]), key(a[-1])
        # orient a so its tail is the shared node, b so its head is
        shared = key(polys[p1][0 if e1 == 0 else -1])
        if key(a[0]) == shared:
            a = a[::-1]
        elif key(a[-1]) != shared:
            continue
        if key(b[-1]) == shared:
            b = b[::-1]
        elif key(b[0]) != shared:
            continue
        merged[r2] = np.vstack([a, b[1:]])
        del merged[r1]
        parent[r1] = r2
    return list(merged.values())


# ---------------------------------------------------------------- driver

def refine_layers(raw: dict[str, list[np.ndarray]],
                  p: RefineParams | None = None) -> dict[str, list[np.ndarray]]:
    """raw: {layer: [raw pixel paths]} -> {layer: [clean polylines]}."""
    p = p or RefineParams()
    runs_by_path: list[list[_Run]] = []
    raw_paths: list[np.ndarray] = []
    path_layer: list[str] = []
    for name, paths in raw.items():
        for path in paths:
            if len(path) < 2:
                continue
            runs = _split_runs(path, p)
            if runs:
                runs_by_path.append(runs)
                raw_paths.append(path)
                path_layer.append(name)

    _snap_runs(runs_by_path, p)

    free: dict[str, list[np.ndarray]] = defaultdict(list)
    fixed: dict[str, list[np.ndarray]] = defaultdict(list)
    for path, runs, name in zip(raw_paths, runs_by_path, path_layer):
        if len(runs) == 1:
            free[name].append(np.array([runs[0].e0, runs[0].e1]))
        else:
            fixed[name].append(_path_geometry(path, runs, p))

    out: dict[str, list[np.ndarray]] = {}
    for name in raw:
        polys = fixed.get(name, []) + _merge_free_segments(free.get(name, []), p)
        _join_endpoints(polys, p)
        out[name] = _chain(polys)
    return out
