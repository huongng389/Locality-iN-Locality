# LNL-TS: Drop-in Traffic-Sign Locality-iN-Locality

## Baseline

The original Colab baseline was reproduced with `LNL_Ti` on GTSRB for 5 epochs.
The measured standard Top-1 accuracy was 97.18%, with 6.08M parameters and
1.25 GMAC.

## Proposed Model

`LNL_TS.py` is designed as a drop-in replacement for the original model import.
The Colab training and evaluation flow remains the same; only the import line is
changed.

Original LNL:

```python
from LNL import LNL_Ti as small
```

Proposed LNL-TS:

```python
from LNL_TS import LNL_Ti as small
```

Original LNL-MoEx:

```python
from LNL_MoEx import LNL_MoEx_Ti as small
```

Proposed LNL-TS MoEx-compatible model:

```python
from LNL_TS import LNL_MoEx_Ti as small
```

## What Changed

The model keeps the original TNT/LNL backbone, classifier interface, and
MoEx-compatible `forward(...)` signature. The main architectural change is the
input stem:

- The original single `7x7 stride-4` convolution is replaced by a two-stage
  local convolutional stem.
- The new stem uses `3x3` convolution, BatchNorm, GELU, and depthwise
  convolution.
- This keeps the same feature-map size expected by TNT while adding stronger
  local edge/color/shape extraction for traffic signs.

Because the classifier remains a normal `Linear(192, 43)` head after the
notebook head replacement cell, the model can still be trained and loaded with
the original Colab code style.

## Why It Helps

Traffic signs are highly local visual objects: border shape, inner symbols,
color regions, and small geometric details matter. A stronger convolutional stem
improves these low-level local features before the transformer blocks process
patch and pixel tokens. This keeps the locality inductive bias of the original
paper while making the first feature extractor better matched to GTSRB.

## Submission Files

- `LNL_TS.py`: proposed model file.
- `lnl_ts_ti_gtsrb_best.pth`: trained weights after running the updated Colab.
- `MODEL_DESCRIPTION.md`: short explanation of the proposed improvement.
