#!/usr/bin/env python3
"""Generate freshwater hosing forcing files for CM2Mc.

The files are read through data_table as

  "ICE", "runoff_add", "runoff_add", "INPUT/<file>.nc", .true., 1.0

runoff_add [kg/m^2/s] is ADDED to the land runoff in the coupler
(flux_exchange.F90, flux_land_to_ice). The field must be on the ocean
T-grid (ongrid = .true.) and the time axis must cover the whole run:
the model aborts if its date is outside the time range of the file.
Between records the model interpolates linearly in time.

Run from the run case folder:  python3 make_hosing_forcing.py [scenario ...]
(no argument: all scenarios). Do not rewrite a file that a running job reads.
"""
import sys
import numpy as np
import netCDF4 as nc

GRID_DIR = "INPUT-backbone"
OUT_DIR = "INPUT-backbone"

# --- scenario ---------------------------------------------------------------
LAT_MIN, LAT_MAX = 50.0, 70.0
ATLANTIC = 2                 # code in basin_mask.nc
ARCTIC = 4                   # code in basin_mask.nc
ARCTIC_LON_MIN, ARCTIC_LON_MAX = -70.0, 20.0   # excludes Hudson Bay and Barents Sea
PEAK_SV = 0.3
RHO_FW = 1000.0              # kg/m^3
DAYS_PER_YEAR = 365          # noleap calendar

# (model year at 1 January, hosing in Sv); the last record is padding
RAMP = [(1, 0.0), (501, PEAK_SV), (1001, 0.0), (1101, 0.0)]
CONST_TEST = [(1, PEAK_SV), (1101, PEAK_SV)]
# open-ended ramp at the same rate as RAMP (identical to it up to year 501)
RAMP_RATE = PEAK_SV / 500.0  # Sv per year
RAMPUP_END_YEAR = 2001
RAMPUP = [(1, 0.0), (RAMPUP_END_YEAR, RAMP_RATE * (RAMPUP_END_YEAR - 1))]
# hosing frozen at the RAMPUP value of year 201 (branch from the year-201 restart)
FREEZE_YEAR = 201
FREEZE_SV = RAMP_RATE * (FREEZE_YEAR - 1)
CONST_FREEZE = [(1, FREEZE_SV), (RAMPUP_END_YEAR, FREEZE_SV)]
# slow open-ended ramp: 0.4 Sv per 2000 years, file extends to year 6001 (1.2 Sv)
SLOW_RATE = 0.4 / 2000.0     # Sv per year
SLOW_END_YEAR = 6001
SLOWRAMP = [(1, 0.0), (SLOW_END_YEAR, SLOW_RATE * (SLOW_END_YEAR - 1))]
# the same slow ramp, held fixed once it reaches SLOW_CAP_SV (year 1001); identical to SLOWRAMP before that
SLOW_CAP_SV = 0.2
SLOW_CAP_YEAR = 1 + round(SLOW_CAP_SV / SLOW_RATE)
SLOWRAMP_CAP = [(1, 0.0), (SLOW_CAP_YEAR, SLOW_CAP_SV), (SLOW_END_YEAR, SLOW_CAP_SV)]
# the capped slow ramp, held for SLOW_HOLD_YEARS, then ramped down at the same rate to 0 (year 2301) and kept at 0;
# identical to SLOWRAMP_CAP up to year 1301
SLOW_HOLD_YEARS = 300
SLOW_DOWN_YEAR = SLOW_CAP_YEAR + SLOW_HOLD_YEARS
SLOW_ZERO_YEAR = SLOW_DOWN_YEAR + round(SLOW_CAP_SV / SLOW_RATE)
SLOWRAMP_UPDOWN = [(1, 0.0), (SLOW_CAP_YEAR, SLOW_CAP_SV), (SLOW_DOWN_YEAR, SLOW_CAP_SV),
                   (SLOW_ZERO_YEAR, 0.0), (SLOW_END_YEAR, 0.0)]


def read_grid():
    with nc.Dataset(f"{GRID_DIR}/ocean_hgrid.nc") as h:
        x = np.asarray(h["x"][:])
        y = np.asarray(h["y"][:])
        dx = np.asarray(h["dx"][:])
        dy = np.asarray(h["dy"][:])
    # T-cell centres and metrics from the supergrid, as MOM5 does (dat = dxt*dyt).
    # The "area" variable of ocean_hgrid.nc is all missing, so it is not used.
    lon = x[1::2, 1::2]
    lat = y[1::2, 1::2]
    area = (dx[1::2, 0::2] + dx[1::2, 1::2]) * (dy[0::2, 1::2] + dy[1::2, 1::2])
    with nc.Dataset(f"{GRID_DIR}/topog.nc") as t:
        wet = np.ma.filled(t["depth"][:], 0.0) > 0.0
    with nc.Dataset(f"{GRID_DIR}/basin_mask.nc") as b:
        basin = np.ma.filled(b["basin_mask"][:], -1.0)
        xt = np.asarray(b["GRID_X_T"][:])
        yt = np.asarray(b["GRID_Y_T"][:])
    return lon, lat, area, wet, basin, xt, yt


def write_file(fname, series, mask, area, lon, lat, xt, yt):
    region_area = area[mask].sum()
    years = np.array([s[0] for s in series])
    sv = np.array([s[1] for s in series])
    days = (years - 1) * DAYS_PER_YEAR

    with nc.Dataset(fname, "w", format="NETCDF3_64BIT_OFFSET") as f:
        f.createDimension("time", None)
        f.createDimension("yt_ocean", yt.size)
        f.createDimension("xt_ocean", xt.size)

        v = f.createVariable("time", "f8", ("time",))
        v.units = "days since 0001-01-01 00:00:00"
        v.calendar = "noleap"
        v.cartesian_axis = "T"
        v.long_name = "time"
        v[:] = days

        v = f.createVariable("yt_ocean", "f8", ("yt_ocean",))
        v.units = "degrees_north"
        v.cartesian_axis = "Y"
        v.long_name = "nominal latitude of T-cell centre"
        v[:] = yt

        v = f.createVariable("xt_ocean", "f8", ("xt_ocean",))
        v.units = "degrees_east"
        v.cartesian_axis = "X"
        v.long_name = "nominal longitude of T-cell centre"
        v[:] = xt

        v = f.createVariable("runoff_add", "f8", ("time", "yt_ocean", "xt_ocean"))
        v.units = "kg/m^2/s"
        v.long_name = "additional liquid runoff (freshwater hosing)"
        flux = sv[:, None, None] * 1.0e6 * RHO_FW / region_area * mask[None, :, :]
        v[:] = flux

        v = f.createVariable("hosing_sv", "f8", ("time",))
        v.units = "Sv"
        v.long_name = "area-integrated hosing"
        v[:] = sv

        for name, data, units in (("hosing_mask", mask.astype("f8"), "1"),
                                  ("area_t", area, "m^2"),
                                  ("geolon_t", lon, "degrees_east"),
                                  ("geolat_t", lat, "degrees_north")):
            v = f.createVariable(name, "f8", ("yt_ocean", "xt_ocean"))
            v.units = units
            v[:] = data

        f.description = (f"Uniform hosing over the Atlantic {LAT_MIN:g}N-{LAT_MAX:g}N "
                         f"(basin_mask == {ATLANTIC}, plus basin_mask == {ARCTIC} for "
                         f"{ARCTIC_LON_MIN:g}E..{ARCTIC_LON_MAX:g}E); "
                         f"linear in time between records")
        f.region_area_m2 = region_area

    print(f"{fname}")
    print(f"  region: {mask.sum()} cells, {region_area:.4e} m^2, "
          f"lon {lon[mask].min():.1f}..{lon[mask].max():.1f}, "
          f"lat {lat[mask].min():.1f}..{lat[mask].max():.1f}")
    for yr, s, fl in zip(years, sv, flux):
        print(f"  year {yr:5d}: {s:.3f} Sv, integral {np.sum(fl * area):.4e} kg/s, "
              f"max {fl.max():.3e} kg/m^2/s")


def main():
    lon, lat, area, wet, basin, xt, yt = read_grid()
    # basin_mask labels the Atlantic only up to ~64N; Davis Strait/Baffin Bay and
    # the Nordic Seas carry the Arctic code, so take those by longitude.
    north = (basin == ARCTIC) & (lon >= ARCTIC_LON_MIN) & (lon <= ARCTIC_LON_MAX)
    mask = ((basin == ATLANTIC) | north) & (lat >= LAT_MIN) & (lat <= LAT_MAX) & wet
    tag = f"NA{LAT_MIN:g}-{LAT_MAX:g}N"
    scenarios = {
        "ramp": (f"hosing_{tag}_ramp{PEAK_SV:g}Sv.nc", RAMP),
        "const_test": (f"hosing_{tag}_const{PEAK_SV:g}Sv_TEST.nc", CONST_TEST),
        "rampup": (f"hosing_{tag}_rampup{RAMP_RATE * 100:g}SvPerCentury.nc", RAMPUP),
        "freeze": (f"hosing_{tag}_const{FREEZE_SV:g}Sv.nc", CONST_FREEZE),
        "slowramp": (f"hosing_{tag}_rampup{SLOW_RATE * 100:g}SvPerCentury.nc", SLOWRAMP),
        "slowramp_cap": (f"hosing_{tag}_rampup{SLOW_RATE * 100:g}SvPerCentury_hold{SLOW_CAP_SV:g}Sv.nc",
                         SLOWRAMP_CAP),
        "slowramp_updown": (f"hosing_{tag}_rampup{SLOW_RATE * 100:g}SvPerCentury_hold{SLOW_CAP_SV:g}Sv"
                            f"_{SLOW_HOLD_YEARS}yr_rampdown.nc", SLOWRAMP_UPDOWN),
    }
    for name in sys.argv[1:] or scenarios:
        fname, series = scenarios[name]
        write_file(f"{OUT_DIR}/{fname}", series, mask, area, lon, lat, xt, yt)

if __name__ == "__main__":
    main()
