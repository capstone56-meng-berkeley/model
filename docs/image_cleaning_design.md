# SEM Image Cleaning Design

## Overview

SEM microstructure images in the dataset carry overlaid annotations added by
the original paper authors: phase labels, scale bars, panel letters, and
annotation circles. These overlays are not part of the microstructure signal
and mislead CNN feature extractors — a VGG16 backbone will allocate filter
responses to "10μm" text just as readily as to grain boundaries. Cleaning must
remove all overlays while preserving the underlying microstructure texture.

This document covers the design of `src/preprocessing/image_cleaner.py` and
the decisions made during iterative evaluation on the 407-image dataset.

---

## Annotation taxonomy

Surveying the full dataset revealed six distinct overlay types:

| Type | Example | Detection approach |
|---|---|---|
| Yellow phase labels | F, M, Ferrite, Martensite | HSV hue range 15–45°, S > 80 |
| Yellow annotation outlines | Diamond/arrow composites | Same HSV range |
| Red annotation circles | Dashed red outlines, "P", "Fq" | HSV hue 0–8° and 165–180°, S > 20 |
| White scale bar + text | "10μm", "5μm" bar | All channels ≥ 235, corners only |
| White panel letters | (a), (b), (a₁) | Same white mask, corners only |
| SEM data strip | Instrument metadata row | Fixed bottom crop |

Black text annotation boxes (e.g. "Untempered martensite" on a dark rectangle)
are not handled — they share the same pixel intensity range as real
microstructure and cannot be isolated by color alone. They affect approximately
3 images in the dataset.

---

## Pipeline

Processing is applied in this order for each image:

```
1. Build per-channel color masks  (yellow, red, white)
2. Union the masks
3. Dilate the union mask          (catches anti-aliased edges)
4. Inpaint masked regions         (Telea algorithm)
5. Crop bottom N%                 (removes scale bar + data strip)
6. Optional resize to output_size
```

The order matters: inpainting before cropping ensures the inpainter has full
image context when filling regions near the bottom edge.

---

## Color masking

### Yellow (phase labels)

HSV is used rather than RGB because yellow is compact in HSV space regardless
of brightness variation across SEM imaging conditions.

```python
H ∈ [15, 45]  (OpenCV 0–180 scale)
S ≥ 80
V ≥ 100
```

These bounds were chosen to include both saturated yellow (pure label text) and
slightly desaturated yellow (anti-aliased glyph edges on a gray background).

### Red (annotation circles and text)

Red wraps around the HSV hue axis, requiring two ranges:

```python
H ∈ [0, 8]   OR  H ∈ [165, 180]
S ≥ 20
V ≥ 100
```

The saturation threshold was lowered from an initial value of 80 to **20**
after evaluation revealed that JPEG compression creates a halo of
low-saturation pink pixels around red text. At `S_min = 80` these halos
survived cleaning and remained visible as pink blotches. At `S_min = 20` they
are caught without introducing false positives on the grayscale microstructure
(verified across all 407 images — zero additional pixels flagged on images with
no red annotations).

### White (scale bar, panel letters)

```python
R ≥ 235  AND  G ≥ 235  AND  B ≥ 235
```

Applied only within a 20% border strip around the image perimeter
(`white_corners_only=True`). This is the critical constraint: some SEM images
contain genuinely bright microstructure features (high-contrast carbide
particles, oxide inclusions) in the image interior. Without the border
restriction, these are incorrectly masked, shifting image contrast and
blurring real features. Scale bars and panel letters are always positioned in
corners; restricting the white mask to the border eliminates all tested
false-positives.

---

## Dilation

After union-masking, the mask is dilated by a 3×3 elliptical structuring
element (`dilate_kernel=3`).

**Why dilate at all:** JPEG compression anti-aliases annotation edges. A glyph
pixel is cleanly yellow at its centre but bleeds into adjacent pixels as
yellow-gray. Without dilation, inpainting fills from these half-yellow
neighbours and produces colour fringing.

**Why 3 and not larger:** Initial evaluation used `dilate_kernel=7`. This
caused overmasking on images with large multi-word labels ("Ferrite" in bold):
the dilated mask grew to cover 20–35% of the image and the inpainter produced
visible blotchy patches. Reducing to 3 brought the worst-case mask coverage
from 35% down to 15%, eliminating all overmasking cases (0 images above 20%,
versus 24 at kernel=7).

---

## Inpainting

OpenCV Telea (`cv2.INPAINT_TELEA`) with `inpaint_radius=7`.

Telea was chosen over Navier-Stokes (`INPAINT_NS`) because it produces sharper
texture reconstruction on the high-frequency grain boundary patterns typical of
SEM images. NS tends to over-smooth.

`inpaint_radius=7` (raised from an initial 5) compensates for the tighter
dilation mask at `kernel=3`: with a smaller mask the inpainter needs a larger
neighbourhood to source texture from.

**Known limitation — sparse outlines:** Telea cannot cleanly fill a dashed
circle outline that has no solid interior. The mask covers only the perimeter
pixels; the inpainter has no same-region context to reconstruct from and
produces a faint smear at the seam. This affects approximately 5 images with
dashed red annotation circles. The smear is low-contrast and visually
unobtrusive, but it is a residual artifact.

---

## Bottom crop

A fixed 10% bottom crop removes the scale bar line and, where present, the SEM
instrument metadata strip (magnification, accelerating voltage, detector label).
Fixed cropping is more reliable than detection-based approaches because the
metadata strip is not present in all images — a detector that triggers on row
brightness would misfire on images where the bottom row is simply dark
microstructure.

The crop is applied after inpainting so the inpainter retains full spatial
context when processing the bottom region.

---

## Configuration

All parameters are exposed through `CleanConfig`:

```python
@dataclass
class CleanConfig:
    crop_bottom_fraction: float = 0.10
    crop_top_fraction: float    = 0.0

    inpaint_yellow: bool        = True
    yellow_h_low: int           = 15
    yellow_h_high: int          = 45
    yellow_s_min: int           = 80
    yellow_v_min: int           = 100

    inpaint_red: bool           = True
    red_s_min: int              = 20    # low — catches JPEG halos around red text
    red_v_min: int              = 100

    inpaint_white: bool         = True
    white_threshold: int        = 235
    white_corners_only: bool    = True  # prevents masking bright microstructure
    corner_fraction: float      = 0.20

    dilate_kernel: int          = 3     # small — prevents overmasking on large labels
    inpaint_radius: int         = 7     # larger radius compensates for tight mask
    inpaint_method: int         = cv2.INPAINT_TELEA

    output_size: Optional[Tuple[int, int]] = None
```

---

## Evaluation results

Evaluated across all 407 images after three iterative parameter fixes:

| Metric | v1 (initial) | v2 (kernel=3, corners-only) | v3 (+ red_s_min=20) |
|---|---|---|---|
| Mean mask coverage | 6.38% | 2.91% | 3.06% |
| Max mask coverage | 35.45% | 14.69% | 14.69% |
| Images with >20% overmasking | 24 | 0 | 0 |
| Red halo pixels caught | No | No | Yes |
| Bright microstructure false positives | ~15 imgs | 0 | 0 |

The three parameter changes that drove the improvement:

1. **`dilate_kernel` 7 → 3** — eliminated all overmasking on large multi-word
   yellow labels. Max mask coverage dropped from 35% to 15%.

2. **`white_corners_only=True`** — eliminated false-positive masking of bright
   carbide/oxide features in image interiors. No real microstructure pixels are
   masked.

3. **`red_s_min` 80 → 20** — caught JPEG compression halos around red text,
   removing residual pink blotches. No false positives introduced on 207 images
   with no red annotations.

---

## Remaining limitations

| Issue | Root cause | Affected images |
|---|---|---|
| Faint seam where dashed red circle was | Telea cannot fill a sparse outline with no interior | ~5 |
| Black annotation text boxes not removed | Same intensity as microstructure; no color signal | ~3 |
| Panel letters partially outside corner strip | `corner_fraction=0.20` clips (a₁)-style letters slightly | ~10 |

The panel letter issue can be addressed by increasing `corner_fraction` to
`0.25`. This was not applied by default because a wider corner strip increases
the risk of catching bright microstructure features near edges, and the
`(a₁)`-style letters are small enough that the inpainter handles the visible
portion adequately.

---

## Files

| Path | Purpose |
|---|---|
| `src/preprocessing/image_cleaner.py` | Core module — `CleanConfig`, `clean_image`, `clean_image_file` |
| `notebooks/image_cleaning_demo.ipynb` | Before/after visualisation, mask breakdown, batch runner |
