# FractalProspect-Net v4
## AI-Based Mineral Prospectivity Mapping — Zimbabwe Archean Greenstone Belt

**Lead Researcher:** Romeo Thando, BSc Applied Mathematics & Computational Science, Midlands State University

**Target Journals:** Ore Geology Reviews · Computers & Geosciences · Minerals (MDPI)

---

### Architecture
Dual-branch patch-based CNN with stochastic deposition theorem encoding.

- **Branch 1 — Spectral** (15×15×35): Multi-scale CNN (3×3, 5×5, 7×7) + ResBlocks + CBAM
- **Branch 2 — Stochastic** (15×15×24): 6 Local Singularity Index maps + 18 fractal maps (D_B + H + Λ)
- **Fusion**: Cross-branch attention gate → MLP (384→256→128→64) → MC Dropout → Binary sigmoid

**Theoretical basis:**
- Theorem 1: Singular Geo-Process (Cheng 2007) — LSI maps
- Theorem 2: Multifractal Self-Similarity (Aguilar-Ayala 2024) — D_B, H, Λ

---

### Data
- **Feature stack**: 35-band GEE asset (S2 + L8, 30m, EPSG:4326, dry-season 2019-2024)
- **Positive labels**: 61 MRDS gold deposits, Zimbabwe
- **Master raster**: 59 bands (35 spectral + 6 LSI + 18 fractal)

---

### Repository Structure
```
data/
  raw/          # Downloaded GeoTIFFs from GEE (gitignored)
  processed/    # LSI maps, fractal maps, 59-band master raster
  samples/      # Training CSVs, patch arrays

notebooks/
  01_gee_export.ipynb         # GEE data pipeline
  02_lsi_computation.ipynb    # Singularity index maps (Theorem 1)
  03_fractal_computation.ipynb # Fractal maps (Theorem 2)
  04_master_raster.ipynb      # Stack to 59-band master raster
  05_patch_extraction.ipynb   # Patch extraction + spatial CV split
  06_relationship_analysis.ipynb # Pearson + MI + variogram + C-A fractal
  07_model_training.ipynb     # FractalProspect-Net v4 training (Colab T4)
  08_evaluation.ipynb         # Validation + blind test + geological overlay
  09_prospectivity_map.ipynb  # Full Zimbabwe inference

src/
  data/         # GEE export, raster loading, patch extraction
  features/     # LSI + fractal computation
  models/       # FractalProspect-Net v4 architecture
  training/     # Training loop, loss, optimiser
  evaluation/   # Metrics, spatial CV, Youden threshold
  visualization/ # SHAP, Grad-CAM, maps

outputs/
  models/       # Saved model weights (gitignored)
  maps/         # Prospectivity + uncertainty rasters
  figures/      # Publication figures
  uncertainty/  # MC Dropout epistemic uncertainty maps
```

---

### Setup
```bash
conda activate mineralprospect
pip install -r requirements.txt
```

### GEE Assets
- Feature stack: `projects/mineral-prospectivity-zim/assets/feature_stack`
- Deposits: `projects/mineral-prospectivity-zim/assets/mrds_zim_gold_clean`

---

*Repo is PRIVATE during development. Will be made public + archived on Zenodo at submission.*
