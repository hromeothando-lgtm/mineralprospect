# FractalProspect-Net v4
## AI-Based Mineral Prospectivity Mapping — Zimbabwe Archean Greenstone Belt

**Lead Researcher:** Romeo Thando, BSc Applied Mathematics & Computational Science, Midlands State University

**Target Journals:** Ore Geology Reviews · Computers & Geosciences · Minerals (MDPI)

---

### Architecture
Two-branch patch-based CNN with stochastic deposition theorem encoding.

- **Branch 1 — Spectral** (15×15×35): Multi-scale CNN (3×3, 5×5, 7×7) + ResBlocks + CBAM
- **Branch 2 — Stochastic** (15×15×24): 6 Local Singularity Index maps + 18 fractal descriptor maps
- **Fusion**: Cross-branch attention gate (2-way, 224-d) → Fusion MLP → MC Dropout →
  Binary sigmoid (High/Low, Maximum Youden Index threshold)

**Theoretical basis:**
- Theorem 1: Singular Geo-Process (Cheng, 2007, *Nonlinear Processes in Geophysics*) — LSI maps
- Theorem 2: Multifractal Self-Similarity (Bergami et al., 2024, *Ore Geology Reviews*) — fractal
  descriptor maps

Note: two earlier design phases were removed. A fourth "Chaos" branch (Spatial Lyapunov
Exponent / FTLE, based on Takens 1981 embedding theory) was dropped — spatial Lyapunov
exponents on a static raster composite cannot satisfy Takens' temporal reconstruction
requirement. A Terrain branch (slope, aspect, curvature, drainage density, TRI derived from
the Copernicus GLO-30 DEM) was also attempted and permanently discontinued after repeated
tool-chain failures (pysheds, WhiteboxTools, richdem, and numba all failed to produce a
working country-scale pipeline). See `src/data/README_terrain.md` for details on the
discontinued terrain branch. The current architecture is locked at two branches.

---

### Data
- **Feature stack**: 35-band GEE asset (S2 + L8, 30m, EPSG:4326, dry-season 2019-2024)
- **Positive labels**: 61 MRDS gold deposits, Zimbabwe
- **Master raster**: 59 bands (35 spectral + 6 LSI + 18 fractal)
