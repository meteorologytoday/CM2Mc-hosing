#!/usr/bin/env python3
"""AMOC strength time series from pik_amoc in *.ocean_year.nc (run from the run folder)."""
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATS, DEPTH_MIN = (26.0, 40.0), 500.0              # nearest grid latitudes; maximum over depth below DEPTH_MIN
COLORS = ("#2a78d6", "#d9622b")                    # one per latitude, fixed order
RAMP_SV_PER_YEAR = 0.4 / 2000.0
HOLD_SV = 0.2                                      # held 1001-1301, then ramped down at the same rate
WINDOW = 11                                        # running-mean length in years
TREND_WINDOWS = (31, 61)                           # lengths of the sliding linear fits for the change rate (thin, thick), in years
# North Atlantic SST region: Atlantic basin (code 2) plus the Arctic-coded Nordic Seas / Baffin Bay
# between SST_ARCTIC_LON (as for the hosing region), SST_LAT_MIN-SST_LAT_MAX; area-weighted mean
BASIN_FILE = "/p/projects/poem/tienyiao/projects/CM2Mc_hosing_proposal/Control/INPUT-backbone/basin_mask.nc"
SST_LAT_MIN, SST_LAT_MAX = 0.0, 70.0
SST_ARCTIC_LON = (-70.0, 20.0)
SST_COLOR = "#3d3c39"                              # neutral: not one of the latitude colours

# Sources in increasing precedence: the crashed segment, the per-processor files of a segment
# still running (ocean_year.nc.NNNN, each holding a band of latitudes), then the combined history/ files.
files = (sorted(glob.glob("history_crashed_*/*.ocean_year.nc")) + sorted(glob.glob("ocean_year.nc.[0-9]*"))
         + sorted(glob.glob("history/*.ocean_year.nc")))
series = [dict() for _ in LATS]
glat = [None] * len(LATS)
for fn in files:
    with nc.Dataset(fn) as f:
        lat = np.asarray(f["yu_ocean"][:])
        dep = np.asarray(f["st_ocean"][:])
        yrs = np.asarray(f["time"][:]) / 365.0 + 1.0
        for k, l in enumerate(LATS):
            j = int(np.abs(lat - l).argmin())
            if abs(lat[j] - l) > 2.0:      # this file's latitude band does not contain l
                continue
            glat[k] = lat[j]
            psi = np.ma.filled(f["pik_amoc"][:, dep >= DEPTH_MIN, j], -np.inf).max(axis=1)
            for y, p in zip(yrs, psi):
                if np.isfinite(p):          # a crashed segment leaves an all-missing last record
                    series[k][round(float(y), 1)] = p
years = np.array(sorted(set.intersection(*[set(d) for d in series])))
amoc = np.array([[series[k][y] for k in range(len(LATS))] for y in years])   # (year, latitude)
hosing = np.interp(years, [1.0, 1001.0, 1301.0, 2301.0], [0.0, HOLD_SV, HOLD_SV, 0.0])   # up, held 1001-1301, down to 0 at 2301

# North Atlantic mean SST. The per-processor files each hold one x-y tile, so their area-weighted sums
# are added up; a group of files (crashed segment / running segment / one history file) overrides earlier ones.
with nc.Dataset(BASIN_FILE) as b:
    basin = np.ma.filled(b["basin_mask"][:], -1.0)
    bx, by = np.asarray(b["GRID_X_T"][:]), np.asarray(b["GRID_Y_T"][:])
AREA_FILE = "/p/projects/poem/tienyiao/projects/CM2Mc_hosing_proposal/test_hosing_200yr/history/0002010101.ocean_year.nc"
with nc.Dataset(AREA_FILE) as a:                 # per-processor files of a running job have area_t = 0
    AREA = np.ma.filled(a["area_t"][:], 0.0)    # (also geolon_t/geolat_t: written only at job end)
    GLON, GLAT = np.asarray(a["geolon_t"][:]), np.asarray(a["geolat_t"][:])
sst_series = {}
groups = ([[f] for f in sorted(glob.glob("history_crashed_*/*.ocean_year.nc"))]
          + [sorted(glob.glob("ocean_year.nc.[0-9]*"))]
          + [[f] for f in sorted(glob.glob("history/*.ocean_year.nc"))])
for group in groups:
    num, den, gyrs = 0.0, 0.0, None
    for fn in group:
        with nc.Dataset(fn) as f:
            ix = [int(np.abs(bx - x).argmin()) for x in np.asarray(f["xt_ocean"][:])]
            iy = [int(np.abs(by - y).argmin()) for y in np.asarray(f["yt_ocean"][:])]
            bas = basin[np.ix_(iy, ix)]
            glon, glat_t = GLON[np.ix_(iy, ix)], GLAT[np.ix_(iy, ix)]
            glon = np.where(glon > 180.0, glon - 360.0, glon)
            region = (((bas == 2) | ((bas == 4) & (glon >= SST_ARCTIC_LON[0]) & (glon <= SST_ARCTIC_LON[1])))
                      & (glat_t >= SST_LAT_MIN) & (glat_t <= SST_LAT_MAX))
            gyrs = np.asarray(f["time"][:]) / 365.0 + 1.0
            sst = f["sst"][:gyrs.size]       # a record may be half-written while the job runs
            w = np.where(region, AREA[np.ix_(iy, ix)], 0.0)[None] * ~np.ma.getmaskarray(sst)
            num = num + (np.ma.filled(sst, 0.0) * w).sum(axis=(1, 2))
            den = den + w.sum(axis=(1, 2))
            gyrs = np.asarray(f["time"][:]) / 365.0 + 1.0
    if gyrs is None:
        continue
    for y, n_, d_ in zip(gyrs, num, den):
        if d_ > 0:
            sst_series[round(float(y), 1)] = n_ / d_
sst_na = np.array([sst_series.get(y, np.nan) for y in years])

INK, MUTED, GRID, BLUE, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, (a1, a3, a4, a2) = plt.subplots(4, 1, figsize=(9, 11), sharex=True, facecolor=SURF,
                                     gridspec_kw={"height_ratios": [3, 2.2, 2.2, 1], "hspace": 0.14})
# change rate: slope of a least-squares line fitted to the annual means in a sliding window
def sliding_trend(window):
    half = window // 2
    x = np.arange(window) - half
    rate = np.array([[np.dot(x, amoc[i - half: i + half + 1, k]) / np.dot(x, x) * 100.0
                      for i in range(half, years.size - half)] for k in range(len(LATS))])
    return years[half: years.size - half], rate
trends = [sliding_trend(w) for w in TREND_WINDOWS]
ytrend, rate = trends[-1]
for k, col in enumerate(COLORS):
    smooth = np.convolve(amoc[:, k], np.ones(WINDOW) / WINDOW, mode="valid")
    ysmooth = years[WINDOW // 2: WINDOW // 2 + smooth.size]
    a1.plot(years, amoc[:, k], color=col, lw=1, alpha=0.35)
    a1.plot(ysmooth, smooth, color=col, lw=2, label=f"{glat[k]:.1f}°N")
a1.set_ylabel("AMOC strength (Sv)")
a1.set_title(f"AMOC strength under slowly ramped North Atlantic hosing (0.02 Sv per century)\n"
             f"maximum of pik_amoc over depth below {DEPTH_MIN:g} m; "
             f"thin: annual mean, thick: {WINDOW}-year running mean",
             loc="left", fontsize=11)
a1.legend(frameon=False, loc="upper right")
a3.axhline(0.0, color=MUTED, lw=1)
for k, col in enumerate(COLORS):
    a3.plot(trends[0][0], trends[0][1][k], color=col, lw=1, alpha=0.35)
    a3.plot(ytrend, rate[k], color=col, lw=2)
a3.set_ylabel("Change rate (Sv per century)")
a3.set_title(f"slope of a sliding linear fit to the annual means; "
             f"thin: {TREND_WINDOWS[0]}-year window, thick: {TREND_WINDOWS[1]}-year window",
             loc="left", fontsize=10, color=MUTED)
ok = np.isfinite(sst_na)
sst_smooth = np.convolve(sst_na[ok], np.ones(WINDOW) / WINDOW, mode="valid")
a4.plot(years[ok], sst_na[ok], color=SST_COLOR, lw=1, alpha=0.35)
a4.plot(years[ok][WINDOW // 2: WINDOW // 2 + sst_smooth.size], sst_smooth, color=SST_COLOR, lw=2)
a4.set_ylabel("SST (°C)")
a4.set_title(f"North Atlantic mean SST, {SST_LAT_MIN:g}–{SST_LAT_MAX:g}°N (area-weighted)",
             loc="left", fontsize=10, color=MUTED)
a2.plot(years, hosing, color=MUTED, lw=2)
a2.set_ylabel("Hosing (Sv)")
a2.set_xlabel("Model year")
a2.set_ylim(0, None)
for a in (a1, a3, a4, a2):
    a.set_facecolor(SURF)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    a.set_xlim(years[0] - 0.5, years[-1] + 0.5)
fig.savefig("amoc_timeseries.png", dpi=150, bbox_inches="tight")
for k in range(len(LATS)):
    print(f"{glat[k]:.1f}N: years {years[0]:.1f}-{years[-1]:.1f}; first 10 yr mean {amoc[:10, k].mean():.2f} Sv, "
          f"last 10 yr mean {amoc[-10:, k].mean():.2f} Sv, min {amoc[:, k].min():.2f}, max {amoc[:, k].max():.2f}")
    i = rate[k].argmin()
    print(f"   {TREND_WINDOWS[-1]}-yr change rate: mean {rate[k].mean():.2f}, steepest {rate[k].min():.2f} Sv/century at year {ytrend[i]:.0f}, "
          f"max {rate[k].max():.2f} at year {ytrend[rate[k].argmax()]:.0f}")
print(f"NA SST {SST_LAT_MIN:g}-{SST_LAT_MAX:g}N: first 10 yr mean {np.nanmean(sst_na[:10]):.2f} C, "
      f"last 10 yr mean {np.nanmean(sst_na[-10:]):.2f} C, missing years {int((~np.isfinite(sst_na)).sum())}")
