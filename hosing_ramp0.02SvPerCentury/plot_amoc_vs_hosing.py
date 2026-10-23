#!/usr/bin/env python3
"""AMOC strength and North Atlantic SST against hosing (hysteresis diagram), run from the run folder.

Panels: AMOC at 26N, AMOC at 40N, North Atlantic mean SST (same region as plot_amoc_timeseries.py).
Thick lines: 11-year running mean, coloured by phase (ramp up / hold / ramp down); dots: annual means.
The faster ramp run (0.06 Sv per century, ../test_hosing_200yr) is drawn in gray for context.
"""
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATS, DEPTH_MIN = (26.0, 40.0), 500.0
WINDOW = 11
# hosing of this run: (model year, Sv), linear in between (same as the forcing file)
HOSING = ([1.0, 1001.0, 1301.0, 2301.0], [0.0, 0.2, 0.2, 0.0])
PHASES = (("ramp up, 0.02 Sv per century", 1.0, 1001.0, "#2a78d6"),
          ("held at 0.2 Sv", 1001.0, 1301.0, "#8a8984"),
          ("ramp down, 0.02 Sv per century", 1301.0, 2301.0, "#d9622b"))
FAST_DIR, FAST_RATE = "../test_hosing_200yr", 0.3 / 500.0
# North Atlantic SST region: Atlantic basin (code 2) plus the Arctic-coded Nordic Seas / Baffin Bay
# between SST_ARCTIC_LON, SST_LAT_MIN-SST_LAT_MAX; area-weighted mean
BASIN_FILE = "/p/projects/poem/tienyiao/projects/CM2Mc_hosing_proposal/Control/INPUT-backbone/basin_mask.nc"
SST_LAT_MIN, SST_LAT_MAX = 0.0, 70.0
SST_ARCTIC_LON = (-70.0, 20.0)

with nc.Dataset(BASIN_FILE) as b:
    BASIN = np.ma.filled(b["basin_mask"][:], -1.0)
    BX, BY = np.asarray(b["GRID_X_T"][:]), np.asarray(b["GRID_Y_T"][:])
AREA_FILE = "/p/projects/poem/tienyiao/projects/CM2Mc_hosing_proposal/test_hosing_200yr/history/0002010101.ocean_year.nc"
with nc.Dataset(AREA_FILE) as a:                 # per-processor files of a running job have area_t = 0
    AREA = np.ma.filled(a["area_t"][:], 0.0)    # (also geolon_t/geolat_t: written only at job end)
    GLON, GLAT = np.asarray(a["geolon_t"][:]), np.asarray(a["geolat_t"][:])


def read_run(d):
    """Annual AMOC per latitude and North Atlantic SST; later file groups (crashed < running < history) override."""
    groups = ([[f] for f in sorted(glob.glob(f"{d}/history_crashed_*/*.ocean_year.nc"))]
              + [sorted(glob.glob(f"{d}/ocean_year.nc.[0-9]*"))]
              + [[f] for f in sorted(glob.glob(f"{d}/history/*.ocean_year.nc"))])
    amoc, glat, sst = [dict() for _ in LATS], [None] * len(LATS), {}
    for group in groups:
        num = den = 0.0
        yrs = None
        for fn in group:
            with nc.Dataset(fn) as f:
                yrs = np.asarray(f["time"][:]) / 365.0 + 1.0
                lat, dep = np.asarray(f["yu_ocean"][:]), np.asarray(f["st_ocean"][:])
                for k, l in enumerate(LATS):
                    j = int(np.abs(lat - l).argmin())
                    if abs(lat[j] - l) <= 2.0:
                        glat[k] = lat[j]
                        psi = np.ma.filled(f["pik_amoc"][:, dep >= DEPTH_MIN, j], -np.inf).max(axis=1)
                        amoc[k].update({round(float(y), 1): p for y, p in zip(yrs, psi) if np.isfinite(p)})
                ix = [int(np.abs(BX - x).argmin()) for x in np.asarray(f["xt_ocean"][:])]
                iy = [int(np.abs(BY - y).argmin()) for y in np.asarray(f["yt_ocean"][:])]
                bas = BASIN[np.ix_(iy, ix)]
                glon, glt = GLON[np.ix_(iy, ix)], GLAT[np.ix_(iy, ix)]
                glon = np.where(glon > 180.0, glon - 360.0, glon)
                region = (((bas == 2) | ((bas == 4) & (glon >= SST_ARCTIC_LON[0]) & (glon <= SST_ARCTIC_LON[1])))
                          & (glt >= SST_LAT_MIN) & (glt <= SST_LAT_MAX))
                s = f["sst"][:yrs.size]          # a record may be half-written while the job runs
                w = np.where(region, AREA[np.ix_(iy, ix)], 0.0)[None] * ~np.ma.getmaskarray(s)
                num = num + (np.ma.filled(s, 0.0) * w).sum(axis=(1, 2))
                den = den + w.sum(axis=(1, 2))
        if yrs is not None:
            sst.update({round(float(y), 1): n / dd for y, n, dd in zip(yrs, num, den) if dd > 0})
    years = np.array(sorted(set(sst).intersection(*[set(a) for a in amoc])))
    vals = np.array([[a[y] for a in amoc] + [sst[y]] for y in years])    # (year, AMOC per latitude + SST)
    return years, vals, glat


def running(x):
    return np.convolve(x, np.ones(WINDOW) / WINDOW, mode="valid")


years, vals, glat = read_run(".")
hos = np.interp(years, *HOSING)
fyears, fvals, _ = read_run(FAST_DIR)
fhos = FAST_RATE * (fyears - 1.0)
h = WINDOW // 2

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURF)
axs[1].sharey(axs[0])
titles = [f"AMOC at {g:.1f}°N" for g in glat] + [f"North Atlantic mean SST, {SST_LAT_MIN:g}–{SST_LAT_MAX:g}°N"]
for k, ax in enumerate(axs):
    ax.plot(fhos[h:-h], running(fvals[:, k]), color="#c9c8c3", lw=1.5, ls="--",
            label=f"faster ramp up, 0.06 Sv per century (years 1–{int(fyears[-1])})")
    hs, vs = hos[h:-h], running(vals[:, k])
    ys = years[h:-h]
    for name, y0, y1, col in PHASES:
        m = (years >= y0) & (years <= y1)
        ax.scatter(hos[m], vals[m, k], s=4, color=col, alpha=0.25, lw=0)
        ms = (ys >= y0) & (ys <= y1)
        if ms.any():
            ax.plot(hs[ms], vs[ms], color=col, lw=2, label=f"{name} (years {int(y0)}–{int(min(y1, years[-1]))})")
    ax.plot(hs[-1], vs[-1], "o", ms=8, color=PHASES[-1][3] if years[-1] > 1301 else MUTED,
            mec=SURF, mew=2, zorder=5)
    ax.annotate(f"year {int(years[-1])}", (hs[-1], vs[-1]), xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                fontsize=9, color=MUTED, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
    ax.set_title(titles[k], loc="left", fontsize=11)
    ax.set_xlabel("Hosing, North Atlantic 50–70°N (Sv)")
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.005, max(fhos[-1], hos.max()) + 0.01)
axs[0].set_ylabel("AMOC strength (Sv)")
axs[2].set_ylabel("SST (°C)")
axs[0].legend(frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle(f"AMOC and North Atlantic SST against freshwater forcing "
             f"(dots: annual means, lines: {WINDOW}-year running mean)", x=0.07, ha="left", fontsize=12)
fig.savefig("amoc_vs_hosing.png", dpi=150, bbox_inches="tight")
print(f"this run: years {years[0]:.1f}-{years[-1]:.1f}, hosing now {hos[-1]:.3f} Sv; "
      f"fast run: years {fyears[0]:.1f}-{fyears[-1]:.1f}, up to {fhos[-1]:.3f} Sv")
print(f"last 10 yr: AMOC26 {vals[-10:, 0].mean():.2f}  AMOC40 {vals[-10:, 1].mean():.2f}  SST {vals[-10:, 2].mean():.2f}")
