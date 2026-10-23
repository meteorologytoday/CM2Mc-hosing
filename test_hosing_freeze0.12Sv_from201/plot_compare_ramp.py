#!/usr/bin/env python3
"""Frozen-hosing branch vs the ramp run: AMOC at 26N/40N and North Atlantic mean SST (run from this folder).

Reads combined history/ files and, for a segment still running, the per-processor ocean_year.nc.NNNN files.
"""
import glob
import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAMP_DIR = "../test_hosing_200yr"
BRANCH_YEAR = 201
YEAR_RANGE = (101, 301)                 # x-axis
LATS, DEPTH_MIN = (26.0, 40.0), 500.0   # AMOC: maximum over depth below DEPTH_MIN at the nearest grid latitude
BASIN_FILE = "/p/projects/poem/tienyiao/projects/CM2Mc_hosing_proposal/Control/INPUT-backbone/basin_mask.nc"
SST_LAT_MIN, SST_LAT_MAX = 0.0, 70.0    # same region as plot_amoc_timeseries.py
SST_ARCTIC_LON = (-70.0, 20.0)
RAMP_SV_PER_YEAR = 0.3 / 500.0
WINDOW = 11

with nc.Dataset(BASIN_FILE) as b:
    BASIN = np.ma.filled(b["basin_mask"][:], -1.0)
    BX, BY = np.asarray(b["GRID_X_T"][:]), np.asarray(b["GRID_Y_T"][:])
AREA_FILE = "/p/projects/poem/tienyiao/projects/CM2Mc_hosing_proposal/test_hosing_200yr/history/0002010101.ocean_year.nc"
with nc.Dataset(AREA_FILE) as a:                 # per-processor files of a running job have area_t = 0
    AREA = np.ma.filled(a["area_t"][:], 0.0)    # (also geolon_t/geolat_t: written only at job end)
    GLON, GLAT = np.asarray(a["geolon_t"][:]), np.asarray(a["geolat_t"][:])


def read_run(d):
    """Per-year AMOC (one per LATS) and North Atlantic SST; later file groups override earlier ones."""
    groups = ([[f] for f in sorted(glob.glob(f"{d}/history_crashed_*/*.ocean_year.nc"))]
              + [sorted(glob.glob(f"{d}/ocean_year.nc.[0-9]*"))]
              + [[f] for f in sorted(glob.glob(f"{d}/history/*.ocean_year.nc"))])
    amoc = [dict() for _ in LATS]
    sst = {}
    for group in groups:
        num = den = 0.0
        yrs = None
        for fn in group:
            with nc.Dataset(fn) as f:
                yrs = np.asarray(f["time"][:]) / 365.0 + 1.0
                lat = np.asarray(f["yu_ocean"][:])
                dep = np.asarray(f["st_ocean"][:])
                for k, l in enumerate(LATS):
                    j = int(np.abs(lat - l).argmin())
                    if abs(lat[j] - l) <= 2.0:
                        psi = np.ma.filled(f["pik_amoc"][:, dep >= DEPTH_MIN, j], -np.inf).max(axis=1)
                        amoc[k].update({round(float(y), 1): p for y, p in zip(yrs, psi) if np.isfinite(p)})
                ix = [int(np.abs(BX - x).argmin()) for x in np.asarray(f["xt_ocean"][:])]
                iy = [int(np.abs(BY - y).argmin()) for y in np.asarray(f["yt_ocean"][:])]
                bas = BASIN[np.ix_(iy, ix)]
                glon, glat = GLON[np.ix_(iy, ix)], GLAT[np.ix_(iy, ix)]
                glon = np.where(glon > 180.0, glon - 360.0, glon)
                region = (((bas == 2) | ((bas == 4) & (glon >= SST_ARCTIC_LON[0]) & (glon <= SST_ARCTIC_LON[1])))
                          & (glat >= SST_LAT_MIN) & (glat <= SST_LAT_MAX))
                s = f["sst"][:yrs.size]          # a record may be half-written while the job runs
                w = np.where(region, AREA[np.ix_(iy, ix)], 0.0)[None] * ~np.ma.getmaskarray(s)
                num = num + (np.ma.filled(s, 0.0) * w).sum(axis=(1, 2))
                den = den + w.sum(axis=(1, 2))
        if yrs is not None:
            sst.update({round(float(y), 1): n / dd for y, n, dd in zip(yrs, num, den) if dd > 0})
    years = np.array(sorted(set(sst).intersection(*[set(a) for a in amoc])))
    return (years, np.array([[a[y] for a in amoc] for y in years]), np.array([sst[y] for y in years]))


def smooth(y, x):
    if x.size < WINDOW:
        return y[:0], x[:0]
    s = np.convolve(x, np.ones(WINDOW) / WINDOW, mode="valid")
    return y[WINDOW // 2: WINDOW // 2 + s.size], s


ramp = read_run(RAMP_DIR)
frz = read_run(".")
frz = tuple(a[frz[0] >= BRANCH_YEAR] for a in frz)
# the branch is identical to the ramp run before BRANCH_YEAR; prepend that history so smoothing is continuous
pre = ramp[0] < BRANCH_YEAR
frz_ext = tuple(np.concatenate([r[pre], b]) for r, b in zip(ramp, frz))

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_RAMP, C_FRZ = "#9a9893", "#2a78d6"    # ramp = context (gray), frozen branch = focus
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axs = plt.subplots(4, 1, figsize=(9, 10.5), sharex=True, facecolor=SURF,
                        gridspec_kw={"height_ratios": [2, 2, 2, 1], "hspace": 0.18})
titles = [f"AMOC at {LATS[0]:g}°N (Sv)", f"AMOC at {LATS[1]:g}°N (Sv)",
          f"North Atlantic mean SST, {SST_LAT_MIN:g}–{SST_LAT_MAX:g}°N (°C)"]
for p, ax in enumerate(axs[:3]):
    for (yr, am, ss), col, lab, full in ((ramp, C_RAMP, "ramp (0.06 Sv per century)", ramp),
                                         (frz, C_FRZ, f"hosing frozen at year {BRANCH_YEAR}", frz_ext)):
        val = am[:, p] if p < 2 else ss
        fval = full[1][:, p] if p < 2 else full[2]
        ax.plot(yr, val, color=col, lw=1, alpha=0.35)
        ys, vs = smooth(full[0], fval)
        keep = ys >= (BRANCH_YEAR if full is frz_ext else 0)
        ax.plot(ys[keep], vs[keep], color=col, lw=2, label=lab)
    ax.axvline(BRANCH_YEAR, color=MUTED, lw=1, ls=":")
    ax.set_title(titles[p], loc="left", fontsize=10, color=MUTED)
axs[0].legend(frameon=False, loc="upper right")
yy = np.arange(YEAR_RANGE[0], YEAR_RANGE[1] + 1)
axs[3].plot(yy, RAMP_SV_PER_YEAR * (yy - 1), color=C_RAMP, lw=2)
yf = yy[yy >= BRANCH_YEAR]
axs[3].plot(yf, np.full(yf.size, RAMP_SV_PER_YEAR * (BRANCH_YEAR - 1)), color=C_FRZ, lw=2)
axs[3].set_title("Hosing (Sv)", loc="left", fontsize=10, color=MUTED)
axs[3].set_xlabel("Model year")
fig.suptitle(f"Frozen-hosing branch vs ramp run (thin: annual mean, thick: {WINDOW}-year running mean)",
             x=0.125, y=0.93, ha="left", fontsize=11)
for ax in axs:
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(*YEAR_RANGE)
fig.savefig("compare_ramp.png", dpi=150, bbox_inches="tight")

print(f"frozen branch: years {frz[0][0]:.1f}-{frz[0][-1]:.1f} ({frz[0].size} yr)")
for name, (yr, am, ss) in (("ramp", ramp), ("frozen", frz)):
    for a, b in ((201, 210), (frz[0][-1] - 9.5, frz[0][-1] + 0.5)):
        m = (yr >= a) & (yr <= b)
        print(f"  {name:6s} years {a:.0f}-{b:.0f}: AMOC26 {am[m, 0].mean():.2f}  AMOC40 {am[m, 1].mean():.2f}  "
              f"SST {ss[m].mean():.2f}")
