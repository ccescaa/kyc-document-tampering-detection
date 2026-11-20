This phase covers the exploratory analysis, structural verification, and initial preprocessing of the document tampering datasets used in the project (**DocTamper** and **Tampered-SROIE**).

---

## 1. Objectives
* Inspect the internal storage structure, key-value schemas, and image formats of the **DocTamper** LMDB archives.
* Analyze **Tampered-SROIE (tSROIE)** annotations and generate pixel-level tampering masks and image crops suitable for model training and evaluation.
* Establish clean, reproducible data pipelines in preparation for baseline modeling in Phase 2.

---

## 2. Deliverables & Notebooks

### `01_doctamper_inspection.ipynb` — DocTamper LMDB Analysis
* **Environment & Dependencies:** Configured LMDB reading support (`lmdb`, `numpy`, `cv2`/`PIL`) to parse read-only database environments.
* **Split Verification:** Audited the available dataset splits (`DocTamperV1-TrainingSet`, `DocTamperV1-TestingSet`, `DocTamperV1-FCD`, `DocTamperV1-SCD`).
* **Schema & Key-Value Decoding:** Inspected LMDB cursor entries and key naming conventions (e.g., `image-000000000`), verifying raw byte decoding into `512x512x3` RGB tensors and checking entry counts against expected sample counts.

### `02_tsroie_masks_and_crops.ipynb` — tSROIE Masks & Crops Generation
* **Annotation Parsing:** Extracted tampered region coordinates/bounding boxes from the Tampered-SROIE dataset.
* **Binary Mask Generation:** Converted region annotations into pixel-level binary ground-truth masks highlighting manipulated areas on receipts/documents.
* **Patch & Crop Extraction:** Generated standardized image crops alongside their corresponding mask crops to handle varying document resolutions and preserve local tampering artifacts.
