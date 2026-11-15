"""Pixel -> world-coordinate calibration for scanned architecture drawings.

A Transform maps drawing pixels (x right, y down) to world meters (x east,
y north) via a similarity transform with reflection (the y-flip):

    X = a*px + b*py + tx
    Y = b*px - a*py + ty

scale (m/px) = hypot(a, b); rotation is encoded in the a/b ratio.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Transform:
    a: float
    b: float
    tx: float
    ty: float

    @property
    def scale(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.b, self.a))

    def to_world(self, px: float, py: float) -> tuple[float, float]:
        return (self.a * px + self.b * py + self.tx,
                self.b * px - self.a * py + self.ty)

    def to_px(self, X: float, Y: float) -> tuple[float, float]:
        det = -(self.a * self.a + self.b * self.b)
        dx, dy = X - self.tx, Y - self.ty
        return ((-self.a * dx - self.b * dy) / det,
                (-self.b * dx + self.a * dy) / det)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"a": self.a, "b": self.b, "tx": self.tx, "ty": self.ty,
             "model": "X=a*px+b*py+tx; Y=b*px-a*py+ty",
             "scale_m_per_px": self.scale,
             "rotation_deg": self.rotation_deg}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Transform":
        d = json.loads(Path(path).read_text())
        return cls(d["a"], d["b"], d["tx"], d["ty"])


def from_scale(m_per_px: float, origin_px: tuple[float, float] = (0.0, 0.0),
               origin_world: tuple[float, float] = (0.0, 0.0)) -> Transform:
    """North-up drawing: pure scale + y-flip, anchored at origin_px."""
    a, b = m_per_px, 0.0
    tx = origin_world[0] - a * origin_px[0]
    ty = origin_world[1] + a * origin_px[1]
    return Transform(a, b, tx, ty)


def from_scalebar(px_len: float, meters: float,
                  origin_px: tuple[float, float] = (0.0, 0.0)) -> Transform:
    """Calibrate from a measured scale bar: px_len pixels == meters."""
    return from_scale(meters / px_len, origin_px)


def from_anchors(pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> Transform:
    """Least-squares fit from >=2 (pixel, world) point pairs.

    pairs: [((px, py), (X, Y)), ...]  Needs 2+ pairs; 3+ recommended.
    """
    if len(pairs) < 2:
        raise ValueError("need at least 2 anchor pairs")
    rows, rhs = [], []
    for (px, py), (X, Y) in pairs:
        rows.append([px, py, 1.0, 0.0]); rhs.append(X)
        rows.append([-py, px, 0.0, 1.0]); rhs.append(Y)
    n = 4
    N = [[sum(rows[r][i] * rows[r][j] for r in range(len(rows))) for j in range(n)]
         for i in range(n)]
    v = [sum(rows[r][i] * rhs[r] for r in range(len(rows))) for i in range(n)]
    for i in range(n):
        N[i] = N[i] + [v[i]]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(N[r][c]))
        N[c], N[piv] = N[piv], N[c]
        N[c] = [x / N[c][c] for x in N[c]]
        for r in range(n):
            if r != c and N[r][c] != 0:
                f = N[r][c]
                N[r] = [x - f * y for x, y in zip(N[r], N[c])]
    a, b, tx, ty = (N[r][n] for r in range(n))
    return Transform(a, b, tx, ty)


def fit_residuals(t: Transform,
                  pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[float]:
    out = []
    for (px, py), (X, Y) in pairs:
        Xc, Yc = t.to_world(px, py)
        out.append(math.hypot(Xc - X, Yc - Y))
    return out
