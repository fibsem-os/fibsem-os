"""Compare full-precision vs 2dp LUT: file size, load time, interpolation accuracy."""

import time
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

FULL_PATH = os.path.join(os.path.dirname(__file__), "..", "scaling_factor_lookup_table.csv")
SMALL_PATH = os.path.join(os.path.dirname(__file__), "..", "scaling_factor_lookup_table_2dp.csv")


def build_interpolator(lut: pd.DataFrame) -> RegularGridInterpolator:
    tilts = np.sort(lut["tilt_deg"].unique())
    depths = np.sort(lut["depth_um"].unique())
    n2s = np.sort(lut["n2"].unique())
    wls = np.sort(lut["wavelength_um"].unique())
    nas = np.sort(lut["NA"].unique())

    sort_cols = ["tilt_deg", "depth_um", "n2", "wavelength_um", "NA"]
    lut_sorted = lut.sort_values(by=sort_cols)
    grid_shape = (len(tilts), len(depths), len(n2s), len(wls), len(nas))
    zeta_grid = lut_sorted["zeta"].values.reshape(grid_shape)

    return RegularGridInterpolator(
        points=(tilts, depths, n2s, wls, nas),
        values=zeta_grid,
        bounds_error=True,
        method="linear",
    )


def load_lut(path: str) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    lut = pd.read_csv(path)
    elapsed = time.perf_counter() - t0
    return lut, elapsed


# ── File sizes ────────────────────────────────────────────────────────────────
full_size_mb = os.path.getsize(FULL_PATH) / 1e6
small_size_mb = os.path.getsize(SMALL_PATH) / 1e6

print("=" * 60)
print("FILE SIZE")
print(f"  Full precision : {full_size_mb:.1f} MB")
print(f"  2dp rounded    : {small_size_mb:.1f} MB")
print(f"  Reduction      : {(1 - small_size_mb / full_size_mb) * 100:.1f}%")

# ── Load times ────────────────────────────────────────────────────────────────
lut_full, t_full = load_lut(FULL_PATH)
lut_small, t_small = load_lut(SMALL_PATH)

print("\nLOAD TIME")
print(f"  Full precision : {t_full:.3f} s")
print(f"  2dp rounded    : {t_small:.3f} s")
print(f"  Speedup        : {t_full / t_small:.2f}x")

# ── Build interpolators ───────────────────────────────────────────────────────
interp_full = build_interpolator(lut_full)
interp_small = build_interpolator(lut_small)

# ── Interpolation accuracy ────────────────────────────────────────────────────
# Sample 10 000 random points within the grid bounds
rng = np.random.default_rng(42)
n = 10_000
query_points = np.column_stack([
    rng.uniform(0, 30, n),       # tilt_deg
    rng.uniform(0.5, 14.5, n),   # depth_um
    rng.uniform(1.22, 1.46, n),  # n2
    rng.uniform(0.42, 0.72, n),  # wavelength_um
    rng.uniform(0.2, 0.9, n),    # NA
])

zeta_full = interp_full(query_points)
zeta_small = interp_small(query_points)
errors = np.abs(zeta_full - zeta_small)

print("\nINTERPOLATION ACCURACY (full vs 2dp, 10 000 random query points)")
print(f"  Max error  : {errors.max():.4f}")
print(f"  Mean error : {errors.mean():.4f}")
print(f"  Std error  : {errors.std():.4f}")
print(f"  % points with error < 0.005 : {(errors < 0.005).mean()*100:.1f}%")
print(f"  % points with error < 0.01  : {(errors < 0.01).mean()*100:.1f}%")

# ── Named test cases from the notebook ───────────────────────────────────────
test_cases = [
    ("Meteor bead focal shift",  15, 4,   1.29, 0.58,  0.8),
    ("Cell targeting (Arctis)",  15, 4,   1.35, 0.515, 0.7),
    ("On-grid, shallow",          0, 2,   1.33, 0.52,  0.8),
    ("High-tilt, deep",          25, 12,  1.38, 0.57,  0.9),
]

print("\nNAMED TEST CASES")
print(f"{'Case':<30} {'Full':>8} {'2dp':>8} {'Error':>8}")
print("-" * 60)
for name, tilt, depth, n2, wl, na in test_cases:
    pt = (tilt, depth, n2, wl, na)
    zf = float(interp_full(pt))
    zs = float(interp_small(pt))
    print(f"{name:<30} {zf:>8.4f} {zs:>8.4f} {abs(zf-zs):>8.4f}")

print("=" * 60)
