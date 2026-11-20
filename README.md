# Do Document Tampering Detectors Generalize?

A university computer vision project (2 people, 6 weeks). We test whether detectors of tampered text in document images still work when the data changes: new document sources, recompression, resizing and blur.

>A detector that scores well on the data it was built for can fail on a different kind of document or a re-saved image, and we measure how much, how fast, and why.

---

## 1. The idea

### The problem

People edit documents: they change an amount on a receipt, a date on an invoice, a number on a form. These edits are often invisible to the eye, because the forger replaces a few characters and the rest of the page looks untouched. Companies that process expense claims, insurance claims, loan applications and identity checks therefore want software that looks at the image of a document and answers two questions:

1. **Has it been edited?**
2. **Where?** 

### Why generalization is the real question

Published detectors report strong numbers on the dataset they were trained on. In practice, documents arrive from many sources, and almost every upload is re-compressed by a messaging app, an email client or a scanner. A detector can look excellent in the lab and fail in production for reasons that have nothing to do with tampering itself. For example, it may have learned that "this JPEG quality means fake", or "this layout means fake".

### What this project does

We do **not** try to build a new state-of-the-art detector. We build a controlled experiment around existing ones:

- take detectors of different kinds (a classical forensic model, a foundation-model model, a content-free baseline),
- evaluate them on data they were trained on and on data that differs in a known, controlled way,
- fix every decision (thresholds, preprocessing) in advance so the comparison is fair,
- explain the failures with evidence.

A careful measurement of *where and why detectors break* is a legitimate result even if the models perform badly.

### Research questions

| # | Question | How we test it |
|---|---|---|
| RQ1 | Does performance survive a change of **document source**? | DocTamper test set vs. its cross-source subsets (FCD, SCD) vs. T-SROIE receipts |
| RQ2 | Does it survive **degradation** (recompression, resize, blur)? | A grid of degraded copies of the test images |
| RQ3 | Is a model's score explained by a **shortcut** (compression traces, file properties, "just find text")? | Content-free baselines and controls |
| RQ4 | Does a threshold chosen on in-distribution data still work on shifted data? | One threshold per model, fixed on validation data and never changed |


---

## 2. Task definition

- **Input:** an image of a document (512x512 pixels in the main dataset).
- **Output:** a map with one value per pixel, "how likely is this pixel edited", plus one image-level score.
- **Ground truth:** a binary *mask*, white where text was really edited and black elsewhere.
- **Primary metric:** how well the predicted map overlaps the true mask (F1 and IoU, see section 5).

---

## 3. Data

### DocTamper (main dataset)

| Split | Size | Role |
|---|---|---|
| TrainingSet | 120,000 | Training |
| TestingSet | 30,000 | In-distribution test |
| FCD | 2,000 | Cross-source test  |
| SCD | 18,000 | Cross-source test  |

FCD and SCD as cross-domain test subsets. 
Everything below was measured on 2,000 random samples per split unless stated otherwise.

| Finding | Value |
|---|---|
| Image size | 512x512, all splits |
| Storage format | LMDB database; image bytes under `image-000000000`, masks under `label-000000000` (0-indexed), plus `num-samples` |
| Pristine (all-black mask) images | **0%** in every split |
| Tampered area, median | 0.73% (train), 0.72% (test), 3.11% (FCD), 0.58% (SCD) |
| Tampered area, 95th percentile | 8.6% (train), 4.1% (test), 7.8% (FCD), 2.7% (SCD) |
| JPEG quantization table | identical in all splits (mean 1.00, i.e. quality 100) |
| Mask values | mostly 0/255, some 0/1 (binarize with `mask > 0`) |
| Non-JPEG files | 26 of 2,000 sampled train images are PNG (excluded from training) |

### T-SROIE (unseen test set, never trained on)

- 986 receipts (626 in the `train` folder, 360 in `test`), JPEG at quality about 95 (quantization-table mean 5.77, identical for every file).
- Receipts are large (median about 812x1,700 pixels, up to 4961x7016).
- COCO-style annotations (4-point boxes). There are two categories: **category 2 is the tampered text** (about 0.96% of image area, about 6 boxes per receipt, boxes on numbers) and category 1 is all other text (about 21.6% of area, about 47 boxes per receipt). The dataset ships without a README, so this was established from area statistics and visual inspection.
- We cut each receipt into 512x512 crops at native resolution (no resizing, offsets multiples of 8 to keep the JPEG block grid): 963 receipts used, 23 skipped (too small), **5,152 crops** (3,383 tampered, 1,769 clean). The "clean" crops are untouched regions of tampered receipts, not truly pristine documents.

### What "no pristine images" means for the project

Every image in both datasets contains at least one edit. So:

- **Pixel-level metrics are primary.** The ~98% of untouched pixels, often untouched text in the same document, are a good set of negatives.
- **Image-level ROC-AUC and false-positive rate cannot be measured on DocTamper.** We report a detection rate only (share of tampered images flagged) and say so explicitly.
- **False alarms** are approximated by pixel-level precision, the rate of untouched pixels flagged, and the clean T-SROIE crops.
- Optionally, a small hand-made set of original/forged pairs gives true pristine documents.

---

## 4. Approach

### Models

| # | Model | What it represents |
|---|---|---|
| 1 | **TruFor** (pretrained, zero-shot; optionally briefly fine-tuned) | Forensic approach: low-level traces such as noise and compression inconsistencies. Trained on natural photos, so poor transfer to documents is plausible and would itself be a finding. |
| 2 | **DINOv2 (frozen) + small trained decoder** | Foundation-model approach: structure and semantics. Only the decoder is trained, with binary cross-entropy + Dice loss (needed because edited pixels are a tiny minority). |
| 3 | **Shortcut diagnostics** (content-free baselines) | Checks whether file-level or "text is present" cues explain scores. Since there is no pristine class, planned designs are: predicting which split an image came from using file-level and simple pixel statistics, and a baseline that marks all text blocks as tampered. |

### Fair-comparison rules

1. **One threshold per model**, chosen once on that model's own in-distribution validation set, then never changed for any other test set.
2. Validation data is separated from training data before any training.
3. Test sets are never used for training or threshold selection.
4. Degraded test sets are always made from the **original** images, never from an already degraded copy.
5. Original JPEG bytes are never re-encoded when exporting data (re-encoding destroys the traces forensic models rely on).
6. The same sampled keys (stored in a shared file with a fixed seed) are used by every experiment.

### Test conditions

| Axis | Conditions |
|---|---|
| Document source | DocTamper TestingSet (in-distribution), FCD, SCD, T-SROIE crops |
| Compression | JPEG recompression at quality 90, 80, 75 and 50 (the first three are in the range the DocTamper authors use for their own robustness tests) |
| Resolution | Downscale by 0.75 and 0.5, then resize back so masks stay aligned (masks use nearest-neighbour) |
| Blur | Gaussian blur at one or two strengths |
| Control | DocTamper test images recompressed once at quality 95, to separate the effect of T-SROIE's compression from its different source and tampering method |

The degradation grid is applied to a fixed 1,000-image subset of the test set to keep compute manageable.

### Metrics

- **Pixel level (primary):** F1 and IoU, both pooled over all pixels and averaged per image, plus pixel-level average precision. Accuracy is never used, because about 98% of pixels are untouched.
- **Stratification:** all pixel metrics are also reported by tampered-area bin (for example below 0.5%, 0.5 to 2%, above 2%), because FCD and SCD differ from training in region size as well as source.
- **Image level (secondary):** detection rate only on DocTamper; false-alarm estimates from clean T-SROIE crops, with the caveat above.
- **Calibration:** gap between "best possible threshold per image" (an upper bound, not a result) and the fixed validation threshold.

---

## 5. Plan and team split

| Week | Focus | Person A | Person B |
|---|---|---|---|
| 1 | Data feasibility and toolchain | DocTamper inspection, metric decision record | T-SROIE masks and crops, TruFor toolchain test |
| 2 | Pipeline | Sampling script, shared key files, raw export | Degradation grid; both write and test metrics code |
| 3 | Forensic baseline | TruFor inference on all conditions (validation first) | Scoring rules, thresholds from validation, metrics |
| 4 | Foundation model and diagnostics | Shortcut diagnostics | DINOv2 + decoder training |
| 5 | Full evaluation | Remaining runs, degradation curves | Master tables, calibration analysis |
| 6 | Analysis and writing | Failure gallery | Report, limitations, slides; both freeze the repo |


---


## 6. Reproducing the work (Kaggle)

1. Attach the DocTamper and Tampered_sroie datasets to a notebook (**Add Input**, paste the dataset URL).
2. Run notebook 01 (dataset inspection), then 02 (T-SROIE masks and crops).
3. Notebook 03 (TruFor): turn Internet on, clone `grip-unina/TruFor`, download the weights into `test_docker/weights/trufor.pth.tar`, run `trufor_test.py` from `test_docker/src`. Use `-gpu -1` if no GPU is enabled, and load the checkpoint with `weights_only=False` on recent PyTorch versions.
4. Files written to `/kaggle/working` disappear when the session resets, so save any exported subsets as a Kaggle dataset.

---

## 7. Known limitations and threats to validity

- **Subsets, not full data.** We use random subsets for compute reasons, and results are single-seed unless repeated.
- **No pristine class.** False-alarm behaviour on genuine documents is only approximated.
- **Domain shift is confounded.** T-SROIE differs from DocTamper in source, tampering method, JPEG quality (about 95 vs. 100), image scale and region size at the same time. The quality-95 control and the area stratification reduce, but do not remove, this.
- **FCD and SCD differ in tampered-region size**, so a score change there is not purely a document-source effect.
- **Masks are boxes**, so pixel-level scores measure overlap with boxes around edited text, not exact edited pixels.
- **TruFor was trained on natural photographs.** Weak zero-shot performance partly reflects the domain, not forensic detection in general.
- **"Clean" T-SROIE crops come from tampered receipts**, not pristine documents.

---



## 8. Team

<Francesca Maltoni>, <name 2>. Course: <Deep Learning for Computer Vision>, <Hong Kong University>.
