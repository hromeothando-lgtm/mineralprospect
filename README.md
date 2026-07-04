# FractalProspect-Net v4
## AI-Based Mineral Prospectivity Mapping — Zimbabwe Archean Greenstone Belt

**Lead Researcher:** Romeo Thando, BSc Applied Mathematics & Computational Science, Midlands State University

**Target Journals:** Ore Geology Reviews · Computers & Geosciences · Minerals (MDPI)

---

### Architecture
Three-branch patch-based CNN with stochastic deposition theorem encoding and structural/topographic proxies.

- **Branch 1 — Spectral** (15×15×35): Multi-scale CNN (3×3, 5×5, 7×7) + ResBlocks + CBAM
- **Branch 2 — Stochastic** (15×15×24): 6 Local Singularity Index maps + 18 fractal maps (D_B + H + Λ)
- **Branch 3 — Terrain** (15×15×9): Slope, sin/cos aspect, profile/plan curvature, drainage density
  (r1/r2/r3), TRI — derived from Copernicus GLO-30 DEM via pysheds
- **Fusion**: Cross-branch attention gate (3-way, 448-d) → Fusion MLP (448→256→128→64) →
  MC Dropout → Binary sigmoid (High/Low, Maximum Youden Index threshold)

**Theoretical basis:**
- Theorem 1: Singular Geo-Process (Cheng, 2007, *Nonlinear Processes in Geophysics*) — LSI maps
- Theorem 2: Multifractal Self-Similarity (Bergami et al., 2024, *Ore Geology Reviews*) — D_B, H, Λ
- Structural/topographic control: drainage density and terrain ruggedness are top-ranked
  proxies for orogenic gold systems, addressed via Branch 3

Note: an earlier design phase included a fourth "Chaos" branch (Spatial Lyapunov Exponent /
FTLE, based on Takens 1981 embedding theory). This branch was permanently removed — spatial
Lyapunov exponents on a static raster composite cannot satisfy Takens' temporal reconstruction
requirement, and Branch 3 (Terrain) was introduced in its place to capture structural control
via geomorphologically grounded, non-chaos-theoretic proxies.

---

### Data
- **Feature stack**: 35-band GEE asset (S2 + L8, 30m, EPSG:4326, dry-season 2019-2024)
- **Positive labels**: 61 MRDS gold deposits, Zimbabwe
- **Master raster**: 68 bands (35 spectral + 6 LSI + 18 fractal + 9 terrain)
