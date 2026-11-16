"""calque — turn scanned architecture drawings into scaled, snappable vectors.

Examples:
  # scale bar measured in the image: 107 px == 10 m, drawing is north-up
  calque trace siteplan.jpg --scalebar 107=10 --dxf siteplan.dxf --svg siteplan.svg

  # two known anchor points (px,py=X,Y in meters), e.g. surveyed corners
  calque trace plan.jpg --anchor "1128,180=66.77,79.92" --anchor "1150,1142=63.40,-13.38" --dxf plan.dxf

  # reuse a saved calibration
  calque trace section.jpg --calib siteplan.calib.json --dxf section.dxf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import calibrate
from .export import to_dxf, to_svg
from .vectorize import VectorizeParams, extract


def _parse_anchor(s: str):
    px_part, world_part = s.split("=")
    px, py = (float(v) for v in px_part.split(","))
    X, Y = (float(v) for v in world_part.split(","))
    return ((px, py), (X, Y))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="calque", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("trace", help="vectorize a drawing to DXF/SVG")
    tr.add_argument("image")
    tr.add_argument("--scalebar", help="PXLEN=METERS measured on the drawing's scale bar")
    tr.add_argument("--scale", type=float, help="meters per pixel, if already known")
    tr.add_argument("--anchor", action="append", default=[],
                    help='"px,py=X,Y" pixel->world pair; give 2+ (repeatable)')
    tr.add_argument("--calib", help="load calibration JSON saved by a previous run")
    tr.add_argument("--dxf", help="output DXF path (meters)")
    tr.add_argument("--svg", help="output SVG path (mm)")
    tr.add_argument("--crop", help="x0,y0,x1,y1 region of interest (px)")
    tr.add_argument("--dark-max", type=int, default=110)
    tr.add_argument("--light-max", type=int, default=190)
    tr.add_argument("--simplify", type=float, default=1.6, help="Douglas-Peucker eps in px")
    tr.add_argument("--no-refine", action="store_true",
                    help="skip line regularization (raw skeleton trace)")
    tr.add_argument("--shade", action="store_true",
                    help="also extract posterized soft-shading fill regions")
    tr.add_argument("--preview", help="write a clean line-drawing preview PNG here")
    tr.add_argument("--save-calib", help="write calibration JSON here")
    tr.add_argument("--overwrite", action="store_true",
                    help="allow replacing existing outputs (never inputs or duplicate outputs)")

    args = ap.parse_args(argv)

    inputs = [Path(args.image)] + ([Path(args.calib)] if args.calib else [])
    outputs = [(name, Path(value)) for name in ("save_calib", "dxf", "svg", "preview")
               if (value := getattr(args, name))]

    def aliases(a, b):
        return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))

    # Preflight every destination before loading, tracing, or writing any file.
    try:
        for i, (name, path) in enumerate(outputs):
            option = "--" + name.replace("_", "-")
            if any(aliases(path, source) for source in inputs):
                tr.error(f"{option} aliases input: {path}")
            if any(aliases(path, other) for _, other in outputs[:i]):
                tr.error(f"duplicate output path for {option}: {path}")
        for name, path in outputs:
            if (path.exists() or path.is_symlink()) and not args.overwrite:
                tr.error(f"output already exists: {path}; use --overwrite to replace it")
    except (OSError, RuntimeError) as exc:
        tr.error(f"cannot check output paths: {exc}")

    if args.calib:
        t = calibrate.Transform.load(args.calib)
    elif len(args.anchor) >= 2:
        pairs = [_parse_anchor(a) for a in args.anchor]
        t = calibrate.from_anchors(pairs)
        res = calibrate.fit_residuals(t, pairs)
        print(f"anchors fit: scale {t.scale:.4f} m/px, rotation {t.rotation_deg:.2f} deg, "
              f"max residual {max(res):.3f} m")
    elif args.scalebar:
        px_len, meters = (float(v) for v in args.scalebar.split("="))
        t = calibrate.from_scalebar(px_len, meters)
        print(f"scalebar: {t.scale:.4f} m/px (north-up assumed)")
    elif args.scale:
        t = calibrate.from_scale(args.scale)
        print(f"scale: {t.scale:.4f} m/px (north-up assumed)")
    else:
        print("calibration required: --scalebar, --scale, --anchor x2, or --calib",
              file=sys.stderr)
        return 2

    p = VectorizeParams(dark_max=args.dark_max, light_max=args.light_max,
                        simplify_eps_px=args.simplify, refine=not args.no_refine,
                        shade_regions=args.shade)
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    layers, style = extract(args.image, p, crop, return_style=True)
    counts = {k: len(v) for k, v in layers.items()}
    print("extracted polylines:", counts)

    if args.save_calib:
        t.save(args.save_calib)
        print("calibration ->", args.save_calib)
    if args.dxf:
        n = to_dxf(layers, t, args.dxf, style=style)
        print(f"DXF -> {args.dxf} ({n} entities, meters, weighted + filled)")
    if args.svg:
        n = to_svg(layers, t, args.svg, style=style)
        print(f"SVG -> {args.svg} ({n} paths, mm, styled like the source)")
    if args.preview:
        import cv2
        from .export import render_line
        bgr = cv2.imread(args.image)
        preview = render_line(layers, style, bgr.shape[:2])
        try:
            written = cv2.imwrite(args.preview, preview)
        except cv2.error:
            written = False
        if not written:
            print(f"cannot write preview: {args.preview}", file=sys.stderr)
            return 2
        print(f"preview -> {args.preview}")
    if not (args.dxf or args.svg):
        print("(no --dxf/--svg given; extraction dry run only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
