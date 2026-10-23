#!/usr/bin/env python3
"""SST maps: mean of the first and last N years in history/, and their difference (run from the run folder).

Plotted on the nominal 1-D grid (xt_ocean, yt_ocean); north of ~65N the tripolar grid is curvilinear,
so Arctic positions are only approximate.
"""
import glob
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

NYEARS = 30

files = sorted(glob.glob("history/*.ocean_year.nc"))
yrs, recs = [], []                      # (file, record) for every model year in history/
for fn in files:
    with nc.Dataset(fn) as f:
        for i, t in enumerate(np.asarray(f["time"][:])):
            yrs.append(t / 365.0 + 1.0)
            recs.append((fn, i))
order = np.argsort(yrs)
yrs = np.array(yrs)[order]
recs = [recs[i] for i in order]
assert np.all(np.diff(yrs) == 1.0), "history/ does not hold a contiguous run of years"


def mean_sst(sel):
    tot, n = 0.0, 0
    for fn in sorted({r[0] for r in sel}):
        idx = [r[1] for r in sel if r[0] == fn]
        with nc.Dataset(fn) as f:
            tot = tot + f["sst"][idx].sum(axis=0)
            n += len(idx)
    return tot / n


first, last = mean_sst(recs[:NYEARS]), mean_sst(recs[-NYEARS:])
diff = last - first
y0, y1 = yrs[:NYEARS], yrs[-NYEARS:]       # mid-year times, e.g. 1.5 = model year 1
with nc.Dataset(files[0]) as f:
    lon, lat = np.asarray(f["xt_ocean"][:]), np.asarray(f["yt_ocean"][:])

INK, MUTED, SURF, LAND = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
proj = ccrs.Robinson(central_longitude=-30)
fig, axs = plt.subplots(3, 1, figsize=(8, 13), facecolor=SURF, subplot_kw={"projection": proj})
panels = [
    (first, f"Mean SST, model years {int(y0[0])}–{int(y0[-1])}", "YlOrRd", (-2, 30), "°C"),
    (last, f"Mean SST, model years {int(y1[0])}–{int(y1[-1])}", "YlOrRd", (-2, 30), "°C"),
    (diff, f"Difference (years {int(y1[0])}–{int(y1[-1])} minus {int(y0[0])}–{int(y0[-1])})", "RdBu_r", None, "°C"),
]
for ax, (fld, title, cmap, lim, unit) in zip(axs, panels):
    if lim is None:
        m = float(np.nanpercentile(np.abs(np.ma.filled(fld, np.nan)), 99.5))
        lim = (-m, m)
    ax.set_facecolor(LAND)                 # land = masked cells
    pm = ax.pcolormesh(lon, lat, fld, cmap=cmap, vmin=lim[0], vmax=lim[1],
                       shading="nearest", transform=ccrs.PlateCarree())
    ax.set_global()
    ax.set_title(title, loc="left", fontsize=11)
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.04,
                      extend="both" if cmap == "RdBu_r" else "neither")
    cb.set_label(unit)
    cb.outline.set_visible(False)
fig.savefig("sst_maps.png", dpi=150, bbox_inches="tight")


def area_mean(fld, f, mask):
    a = np.ma.filled(f["area_t"][:], 0.0) * mask * ~np.ma.getmaskarray(fld)
    return float((np.ma.filled(fld, 0.0) * a).sum() / a.sum())


with nc.Dataset(files[0]) as f:
    glat = np.asarray(f["geolat_t"][:])
    for name, m in (("global", np.ones_like(glat)), ("20N-70N", (glat >= 20) & (glat <= 70)),
                    ("southern 70S-20S", (glat >= -70) & (glat <= -20))):
        print(f"{name:>18}: first {area_mean(first, f, m):6.2f} C, last {area_mean(last, f, m):6.2f} C, "
              f"change {area_mean(diff, f, m):+.2f} C")
d = np.ma.filled(diff, np.nan)
j, i = np.unravel_index(np.nanargmin(d), d.shape)
print(f"strongest cooling {d[j, i]:.2f} C at lon {lon[i]:.1f} lat {lat[j]:.1f}; "
      f"strongest warming {np.nanmax(d):.2f} C")
