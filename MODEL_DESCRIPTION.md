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
- The stem now expands to `2x` hidden channels before projecting back to the
  TNT inner-token dimension.
- The classifier feature is the average of the CLS token and the mean patch
  token feature, which is usually more stable for centered traffic signs than
  relying on CLS alone.
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

## Latest Notebook Result

`tried_to_improve/improve_4_.ipynb` reports the current best result from the
non-MoEx `LNL_TS.LNL_Ti` path:

- GTSRB Top-1: 99.24%
- FGSM robust accuracy: 89.15%
- PGD robust accuracy: 54.79%
- Parameters: 6.08M
- Complexity: 1.25 GMac

The MoEx path in the same notebook reports lower clean accuracy, so the next
99.5%+ attempt should fine-tune from the non-MoEx LNL-TS checkpoint first.

## 99.5%+ Fine-Tuning Recipe

Continue from `pretrained/lnl_ts_ti_gtsrb.pth` with mild augmentation, low
learning rate, EMA, and TTA-based checkpoint selection:

```bash
python train_gtsrb_lnl_ts.py \
  --train-dir ./data/GTSRB/Final_Training/Images \
  --test-dir ./data/GTSRB/test \
  --resume-weights pretrained/lnl_ts_ti_gtsrb.pth \
  --epochs 20 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 2e-2 \
  --aug-level finetune \
  --moex-prob 0.0 \
  --mixup-alpha 0.0 \
  --drop-rate 0.0 \
  --drop-path-rate 0.02 \
  --select-tta
```

For the final number, evaluate both the best EMA checkpoint and the averaged
checkpoint with deterministic multi-scale TTA:

```bash
python eval_gtsrb_lnl_ts.py --weights checkpoints/lnl_ts_ti_gtsrb_best.pth --tta
python eval_gtsrb_lnl_ts.py --weights checkpoints/lnl_ts_ti_gtsrb_topk_avg.pth --tta
```

Use whichever checkpoint reports the higher test accuracy.

The same recipe has also been appended to
`tried_to_improve/improve_4_.ipynb` as a new fine-tuning section that saves
`pretrained/lnl_ts_ti_gtsrb_finetuned.pth`.

## Submission Files

- `LNL_TS.py`: proposed model file.
- `lnl_ts_ti_gtsrb_best.pth`: trained weights after running the updated Colab.
- `MODEL_DESCRIPTION.md`: short explanation of the proposed improvement.
