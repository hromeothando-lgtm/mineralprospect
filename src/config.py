# ================================================================
# FractalProspect-Net v4 — Central Configuration
# All hyperparameters and paths defined here.
# Architecture decisions are LOCKED — do not change without cause.
# ================================================================

# ── Paths ────────────────────────────────────────────────────────
GEE_PROJECT        = 'mineral-prospectivity-zim'
FEATURE_STACK_ASSET = 'projects/mineral-prospectivity-zim/assets/feature_stack'
DEPOSITS_ASSET     = 'projects/mineral-prospectivity-zim/assets/mrds_zim_gold_clean'

RAW_RASTER_PATH    = 'data/raw/feature_stack_35band.tif'
MASTER_RASTER_PATH = 'data/processed/master_raster_59band.tif'
LSI_RASTER_PATH    = 'data/processed/lsi_maps_6band.tif'
FRACTAL_RASTER_PATH = 'data/processed/fractal_maps_18band.tif'
SAMPLES_PATH       = 'data/samples/training_samples.csv'
PATCHES_PATH       = 'data/samples/patches.npz'

# ── Raster ───────────────────────────────────────────────────────
SCALE              = 30       # metres per pixel
EPSG               = 4326     # WGS84 — FINAL, do not revisit

# ── Band configuration ───────────────────────────────────────────
# 35 GEE spectral bands (confirmed in asset)
SPECTRAL_BANDS = [
    'S2_IOI', 'S2_FII', 'S2_CI', 'S2_CarbI',
    'S2_IOI_nd', 'S2_FII_nd', 'S2_CI_nd',
    'S2_NDVI', 'S2_NDWI',
    'Blue', 'Green', 'Red', 'NIR',
    'RE1', 'RE2', 'RE3', 'RE4', 'SWIR1', 'SWIR2',
    'L8_IOI', 'L8_FII', 'L8_CI', 'L8_CarbI',
    'L8_IOI_nd', 'L8_FII_nd', 'L8_CI_nd',
    'L8_NDVI', 'L8_NDWI',
    'L8_Blue', 'L8_Green', 'L8_Red', 'L8_NIR',
    'L8_SWIR1', 'L8_SWIR2', 'L8_TIR'
]

# 6 pathfinder bands used for LSI + fractal computation
PATHFINDER_BANDS = [
    'S2_IOI_nd', 'S2_FII_nd', 'S2_CI_nd',   # indices 4,5,6 (0-indexed)
    'L8_IOI_nd', 'L8_FII_nd', 'L8_CI_nd'    # indices 23,24,25 (0-indexed)
]
PATHFINDER_INDICES = [4, 5, 6, 23, 24, 25]  # 0-indexed in spectral stack

N_SPECTRAL  = 35   # Branch 1 channels
N_LSI       = 6    # Branch 2A channels (Theorem 1)
N_FRACTAL   = 18   # Branch 2B channels (Theorem 2)
N_STOCHASTIC = N_LSI + N_FRACTAL  # 24 — Branch 2 total
N_TOTAL     = N_SPECTRAL + N_STOCHASTIC  # 59 — master raster

# ── Patch extraction ─────────────────────────────────────────────
PATCH_SIZE         = 15       # pixels — 450m x 450m at 30m
PATCH_SIZE_OPTIONS = [9, 15, 21]  # Optuna search space

# ── Training data ────────────────────────────────────────────────
N_POSITIVES        = 61
BLIND_TEST_SIZE    = 10       # held-out deposits, never seen during training
BUFFER_RADIUS      = 8500     # metres around deposits for patch sampling
NEG_POS_RATIO      = 1        # 1:1 target

# ── Spatial CV ───────────────────────────────────────────────────
N_OUTER_FOLDS      = 5        # spatial blocking CV outer loop
N_OPTUNA_TRIALS    = 50       # HPO trials per outer fold
RANDOM_SEED        = 42

# ── Architecture ─────────────────────────────────────────────────
SPECTRAL_BRANCH_DIM  = 256    # GAP output dim for Branch 1
STOCHASTIC_BRANCH_DIM = 128   # GAP output dim for Branch 2
FUSION_DIM           = SPECTRAL_BRANCH_DIM + STOCHASTIC_BRANCH_DIM  # 384
MLP_DIMS             = [256, 128, 64]  # Fusion MLP hidden dims

# ── Training ─────────────────────────────────────────────────────
BATCH_SIZE         = 32
MAX_EPOCHS         = 100
LEARNING_RATE      = 1e-3     # Optuna range: 1e-4 to 1e-2
WEIGHT_DECAY       = 1e-4     # Optuna range: 1e-5 to 1e-2
DROPOUT_RATE       = 0.3      # Optuna range: 0.1 to 0.5
FOCAL_GAMMA        = 2.0      # Optuna range: 0.5 to 4.0
FOCAL_ALPHA        = None     # computed from class frequencies
SINGULARITY_BETA   = 0.1      # physics-informed loss weight, Optuna: 0.0 to 0.5

# ── MC Dropout uncertainty ───────────────────────────────────────
MC_DROPOUT_T       = 50       # forward passes at inference

# ── LSI computation ──────────────────────────────────────────────
LSI_RADII          = [5, 15, 50]  # pixels — 150m, 450m, 1500m (3 scales)

# ── Fractal computation ──────────────────────────────────────────
FRACTAL_WINDOW     = 15       # sliding window size for D_B, H, Λ

assert len(SPECTRAL_BANDS) == N_SPECTRAL, \
    f"SPECTRAL_BANDS count mismatch: {len(SPECTRAL_BANDS)} != {N_SPECTRAL}"
assert len(PATHFINDER_BANDS) == N_LSI, \
    f"PATHFINDER_BANDS count mismatch: {len(PATHFINDER_BANDS)} != {N_LSI}"
assert N_STOCHASTIC == 24, f"Stochastic bands: {N_STOCHASTIC} != 24"
assert N_TOTAL == 59, f"Total bands: {N_TOTAL} != 59"
