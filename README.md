# Calque

Calque is a Python CLI for converting scanned architectural drawings into scaled,
layered DXF and SVG linework. You supply a raster plan or section and a known
scale or point coordinates. It traces strokes and colored regions, regularizes
line segments, and exports geometry for further drawing or CAD work. Calibration,
vectorization, refinement and export are also available as Python functions.

## Install

From this source folder, use Python 3.11 for the reproducible setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
python -m pip install --no-build-isolation --no-deps .
calque --help
calque trace --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead.
The package declares OpenCV (headless, with contrib modules), NumPy, ezdxf and
svgwrite. The `test` extra adds pytest. This copy was tested on macOS with
Python 3.11; Windows, Linux and other Python versions were not tested locally.
The GitHub test workflow is configured for Python 3.11 on Linux but has not
been run for this copy.

`requirements.lock` pins and hashes the runtime, test and build dependencies,
including their transitive dependencies. Install it first; disabling build
isolation and dependency resolution for the local package prevents a second,
unlocked dependency download. The wheel-only install fails if a compatible
locked wheel is unavailable. The package still declares Python 3.11 or newer;
other Python versions need separate validation.

To refresh the lock with an installed `uv`, keep `requirements-build.in` aligned
with `build-system.requires` in `pyproject.toml`, then run:

```bash
uv pip compile pyproject.toml requirements-build.in --extra test --universal \
  --python-version 3.11 --generate-hashes --no-annotate -o requirements.lock
```

Review the version/hash changes and rerun the tests in a fresh locked environment
before accepting an update.

## Worked example

Supply `plan.png`, a drawing whose measured scale reference is **200 pixels for
10 meters**. Substitute your own measured values; Calque does not read labels or
find a scale bar automatically. Commands run from the source folder after activation.

```bash
mkdir -p out
calque trace plan.png --scalebar 200=10 \
  --dxf out/plan.dxf --svg out/plan.svg \
  --save-calib out/plan.calib.json --preview out/plan-preview.png
```

DXF coordinates are in meters, with drawing pixel Y flipped to world Y-up. SVG
uses millimeters, translates the drawing to its bounds, and uses SVG's Y-down
coordinates. These are full-size dimensions, not a paper-scale sheet. The PNG
preview uses the original image dimensions and omits region fills and shading.
Create output directories before running the CLI.
Output paths must be distinct and cannot alias the source image or loaded
calibration, including through symlinks or hard links. New output names differing
only in case also count as duplicates on case-insensitive filesystems. Existing
output files require `--overwrite`; that flag never permits input aliases or
duplicate outputs.

If you know pixel-to-world correspondences instead, repeat `--anchor` at least
twice. This example maps pixels to meters at 0.05 m/px:

```bash
calque trace plan.png --anchor '60,60=3,-3' --anchor '360,60=18,-3' \
  --anchor '60,260=3,-13' --dxf out/anchored.dxf
```

The fit reports scale, rotation and maximum anchor residual. Check that residual
and measure a known wall in the export. Reuse a calibration only for an image
with the same pixel scale, origin and rotation:

```bash
calque trace plan.png --calib out/plan.calib.json --svg out/reused.svg
```

## CLI reference

There is one subcommand: `calque trace IMAGE`. The input must be a raster that
OpenCV can decode; PDF input is not implemented.

| Option | Meaning |
| --- | --- |
| `--scalebar PXLEN=METERS` | Known measured pixel length and real length; assumes no rotation. |
| `--scale FLOAT` | Known meters per pixel; assumes no rotation. |
| `--anchor 'px,py=X,Y'` | Repeat for at least two pixel/world pairs; fits a similarity transform with Y reflection. |
| `--calib FILE` | Load saved calibration JSON. |
| `--save-calib FILE` | Write calibration JSON. |
| `--dxf FILE` | Write layered DXF in meters. |
| `--svg FILE` | Write grouped SVG in millimeters. |
| `--preview FILE` | Write a line-only PNG preview. |
| `--overwrite` | Allow replacing existing output files; inputs and duplicate output paths remain protected. |
| `--crop x0,y0,x1,y1` | Trace this pixel rectangle; output retains full-image coordinates. |
| `--dark-max INT` | Heavy stroke gray threshold; default 110. |
| `--light-max INT` | Maximum gray value for pale strokes; default 190. |
| `--simplify FLOAT` | Douglas–Peucker epsilon in pixels; default 1.6. Used for raw traces and contour fallback; refinement has its own tolerances. |
| `--no-refine` | Disable line regularization; retain the simplified skeleton trace. |
| `--shade` | Also extract two bands of soft-shading regions. |
| `-h`, `--help` | Show help for the program or `trace`. |

At least one calibration method is required. If several are supplied, precedence
is `--calib`, then two or more anchors, then `--scalebar`, then `--scale`.
Without `--dxf` or `--svg`, extraction still runs and requested calibration or
preview files are still written.

## Output layers

Classification uses tone, color and shape heuristics, not architectural meaning.
A dark text stroke can become heavy linework just as a wall can.

| DXF layer | SVG group | Meaning |
| --- | --- | --- |
| `CALQUE_HEAVY` | `calque-heavy` | Dark strokes, commonly walls, cut edges and outlines. |
| `CALQUE_LIGHT` | `calque-light` | Pale strokes, commonly hatch, furniture and tile lines. |
| `CALQUE_VEG` | `calque-green` | Outlines of green regions; treated as vegetation. |
| `CALQUE_ACCENT` | `calque-accent` | Other colored regions. |
| `CALQUE_POTS` | `calque-pots` | Ellipse symbols inferred from small white holes in green regions. |
| `CALQUE_SHADE_DK`, `CALQUE_SHADE_LT` | `calque-shade_dark`, `calque-shade_light` | Optional posterized shading. |
| Region layer plus `_FILL` | Part of the corresponding region group | Solid DXF hatches for region fills. |

Absent features need not produce entities. Lines are polylines; pot symbols are
ellipses when detected. Styled exports carry estimated stroke weights, region
fill colors and holes. SVG also preserves measured grayscale stroke tones;
DXF strokes use their layer colors rather than measured tones. The DXF and SVG
are editable geometry, not a building model with wall or room objects.

## Tests and limits

It has been exercised on synthetic drawings in tests and on the owner's own
reference drawings, which are not distributed. The retained reference evidence
covers a published site plan and notes about a colored axonometric drawing;
those drawings were not rerun while preparing this copy. Synthetic tests cover
orthogonal walls, a diagonal, pale strokes, colored regions and a known scale
reference. They parse the DXF and SVG, require populated layers, check a known
15 m wall within **0.30 m (2%)**, and check calibration at 0.05 m/px within
**1e-12 m/px**. Tests also exercise all four CLI calibration routes, saved
calibration, previews, evaluation-script input errors and synthetic reference JSON.

```bash
python -m pytest -q -rxX
```

Two strict expected failures record current limitations; neither is a skip:

- An 8 px gap in the raster becomes 12 px after thinning. The default refinement
  bridges at most 8 px between traced endpoints, so this wall remains split.
- The synthetic styled preview has visual MAE 33.76, above the evaluation
  threshold of 24. Line recall is 0.998, precision 1.0 and green-region IoU
  0.9718, but the stroke/region appearance still differs. The threshold was
  not loosened. A future fix that passes an expected-failure test will fail
  the suite as an unexpected pass until the marker is removed.

To reproduce the standalone synthetic run and check its dimensions:

```bash
python tests/synthetic_plan.py out/synthetic.png
calque trace out/synthetic.png --scalebar 200=10 \
  --dxf out/synthetic.dxf --svg out/synthetic.svg \
  --save-calib out/synthetic.calib.json --preview out/synthetic-preview.png
python tests/check_outputs.py out/synthetic.dxf out/synthetic.svg out/synthetic.calib.json
```

Scale calibration cannot correct perspective, paper distortion or unequal X/Y
scales. OCR, automatic dimension recognition, PDF rasterization and 3D model
generation are not implemented. Text, noisy scans and dense hatch may produce
unwanted geometry; short or faint strokes may disappear. Color, canopy, pot and
roof-rim reconstruction are heuristics. The synthetic tests do not establish
accuracy on arbitrary drawings, sections or photographs. Check exported lengths
and overlays before relying on them. Native CAD/illustration application imports
were not checked for this copy. Some malformed CLI values or unwritable output
paths still raise Python errors.

## Optional reference evaluations

`evals/results_lvv.json` is a historical La Vallée Verte site-plan evaluation against building footprints; the source drawings are not distributed and the result was not reproduced for this copy.

Both scripts can be imported without reference files. Running them without
required arguments or with missing input files gives an error and nonzero exit.
Supply your own files:

```bash
python evals/eval_fidelity.py plan.png --light-max 205 --debug out/fidelity.jpg
python evals/eval_lvv.py plan.png geometry.json registration.json --output out/geometry-results.json
```

The fidelity script compares a styled render with the raster and returns 1 if
its thresholds fail (including on the synthetic plan above). Debug images are
optional. The site-plan evaluation is case-specific: it selects building IDs
`1064543187`, `1064543188`, `1064543189` in `{"osm": [{"id": ..., "pts": [[X, Y], ...]}]}`,
with coordinates in local meters. Registration JSON needs `theta_deg`, `scale`
(meters/pixel), `crater_px` and `crater_m` (two-coordinate centers). Its fixed
center/radius separate outer vertices from roof/terrace arcs. It performs three
best-fit iterations before scoring; this is not an independent survey-accuracy
measurement, and the size of the fitted correction is reported but not gated.
No output JSON is written unless `--output` is given, so the historical results
are not overwritten by default.

## Agent skill

`skills/calque/SKILL.md` contains an optional workflow for agents that read skill
files. It uses the same installed CLI; no agent integration is required to use
the package.

## License

MIT; see [LICENSE](LICENSE).
