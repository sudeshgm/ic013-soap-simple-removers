# IC013: failure report

**Status: KILLED (blocked by an unpublished setting in the SOAP paper)**, 2026-10-25.

This is *not* the plan's scientific KILL outcome. Whether SOAP's choice of directions beats simple removers was never tested. The project ended at the reproduction check, before any comparison.

## 1. What we asked
SOAP (arXiv 2604.00172) cleans up vision-transformer features by removing a few chosen directions. We wanted to know if SOAP's way of choosing those directions beats simple removers: dropping the top principal components (PCs, the directions of biggest variation), random directions, or directions found from pure-noise images (INSID3).

## 2. Exp -1: does SOAP just pick the obvious directions?
We fitted SOAP's projector ourselves on 15,000 ImageNet training images (15 per class, seed 0) for 5 models: DINOv2-B/14 (registers), iBOT-B/16, MAE-B/16, CAPI-L/14 and DINOv3-B/16.
- SOAP removes only **2 directions** per model with its default settings.
- Overlap with the top-2 PCs: 50% for DINOv2 and DINOv3, and 0% for iBOT, MAE and CAPI.
- Overlap with the noise-image directions (mean cos²) is at most 0.38 (MAE) and 0.28 (DINOv3), and about chance (~0.003) for the rest.
- The pre-set stop rule (at least 80% overlap on 4 of 5 models) was not met, so we continued.

Raw numbers: [results/exp1_results.json](results/exp1_results.json)

## 3. Reproduction check (ECSSD, 1,000 images, TokenCut, bilateral solver on)
| Run | Ours maxFβ | Paper maxFβ |
|---|---|---|
| DINOv2 without SOAP | 71.96 | 71.319 |
| DINOv2 with SOAP (headline "Corrected") | 80.80 | 80.633 |
| MAE without SOAP | 80.62 | 79.952 |
| MAE with SOAP (simple selection) | 80.61 | 80.525 |
| MAE with SOAP (headline "Corrected") | not reproducible | 82.094 |

DINOv2 (both runs) and MAE without SOAP matched within 1 point. MAE with SOAP could not be reproduced:
- Our first fit used the wrong feature type: it was fitted on output tokens but applied to key features. A refit was prepared and then cancelled when the project stopped.
- More importantly, the paper's headline "Corrected" numbers depend on a "guide" PC and a sign flip that are chosen **by hand for each model**. The code takes them as inputs, and the notebook hard-codes them for DINOv2 (-2, flipped) and CAPI (-1). No value or rule is given for MAE. Our own attempt at an automatic rule did not recover the authors' choices (see [GUIDE_RULE.md](GUIDE_RULE.md)).

Raw numbers: [results/block2_ecssd.json](results/block2_ecssd.json)

## 4. Why we stopped
The gate we set in advance required both models to reproduce within 1 point, with no hand-tuning. MAE's headline setting can't be set by any stated rule, and we would not hand-tune it to force a match. So the gate failed. Because the blocker sits in the paper and is outside our control, the project was killed.

## 5. Side findings
- **Repo issues in dsb-ifi/soap @06ede71:**
  - `get_data()` is an empty stub in both main scripts.
  - CUDA is hard-coded.
  - Their "dinov2_base" is actually the registers model (`vit_base_patch14_reg4_dinov2`).
  - The fit transform crops randomly because of the default aspect-ratio jitter.
  - The README promises "code and weights", but no weights are released.
- **Paper inconsistency:** Table E.1 reports 79.803 maxFβ for DINOv2, while Table 2 reports 71.319 for the same setting. E.1 looks like it was run with the guide.
- **Where the gain comes from:** on DINOv2, SOAP's projection alone adds about 0.1 points (72.0 to 72.1). The big jump to 80.8 comes from the hand-picked guide PC used to choose the foreground, not from removing directions.

## 6. What would reopen this
The authors publishing the guide PC and flip for MAE and every other model, or a written rule for choosing them that reproduces their DINOv2 and CAPI choices.

Datasets, features and weights are not included (non-commercial licences).
