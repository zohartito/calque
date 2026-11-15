---
name: calque
description: Convert a user-supplied raster architectural plan or section into calibrated DXF and SVG linework. Use for tracing a drawing to CAD, calibrating pixels to meters, or preparing editable linework from a scan. Requires the Calque Python package and a known scale or pixel-to-world anchors.
---

# Calque: raster drawings to scaled linework

Use an environment with the package installed. From the repository root,
`python -m pip install '.[test]'` installs the CLI and test dependencies.
Read `README.md` for options, layer names, known limits and expected failures.
Commands below run from that root with the environment activated.

1. Obtain the raster image and a known length or at least two pixel-to-world
   point correspondences. Do not infer a real-world scale from appearance.
2. Inspect a crop to measure the scale reference in pixels. Scale-bar and direct
   scale modes assume no rotation; use anchors when rotation matters.
3. Trace, save the calibration, and export both geometry and a preview. Replace
   the example scale with the user's measured values:

   ```bash
   mkdir -p out
   calque trace plan.png --scalebar 200=10 \
     --dxf out/plan.dxf --svg out/plan.svg \
     --save-calib out/plan.calib.json --preview out/plan-preview.png
   ```

4. Check the line preview against the input and measure a known wall in the
   export. DXF uses meters; SVG uses millimeters. Dark and pale layers are
   tone classes, not verified walls or furniture. Region fills are separate
   DXF `_FILL` layers and can be hidden for line-only work.
5. Adjust `--dark-max` and `--light-max` only after inspecting lost or unwanted
   strokes. `--no-refine` helps compare raw traces with regularized output.
   `--shade` adds two shading bands. Text is not recognized or removed.
6. Reuse `--calib` only when image scale, origin and rotation match. Do not
   assume drawings in a set share a calibration.

For a pipeline change, run `python -m pytest -q -rxX`. The synthetic tests need
no external images and include known failures for raster-gap bridging and styled
appearance. Do not describe an expected failure as a passing check.

For a supplied drawing, optionally run:

```bash
python evals/eval_fidelity.py plan.png --debug out/fidelity.jpg
```

Inspect the lost-ink overlay, styled render and visual-difference image as well
as the metrics. The separate `eval_lvv.py` is a case-specific reference check,
not a general survey-registration tool. Its original drawings are not included.
Keep successful file export, measured accuracy and native CAD import as separate
claims; report only the checks actually performed.
