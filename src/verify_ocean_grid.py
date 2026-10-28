import xarray as xr
import numpy as np
from pathlib import Path

# How MOM5 load grid: ocean_grids.F90


def test_array_equality(a, b, atol=1e-5, rtol=0.0):
    
    if a.shape != b.shape:
        raise ValueError("Shapes are not consistent.")
    
    exactly_same = np.all(a==b)
    all_close = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return exactly_same, all_close
        

def verify_history_and_hgrid_files(
    ocean_hgrid_file,
    ocean_history_file,
    atol: float = 1e-3,
    rtol: float = 1e-5,
):
    print(f"Check hgrid file {ocean_hgrid_file!r} against history file {ocean_hgrid_file!r}")

    ds_hgrid = xr.open_dataset(ocean_hgrid_file)
    ds_hist  = xr.open_dataset(ocean_history_file)
    
    print("ds_hgrid:")
    print(ds_hgrid)
    
    print("ds_hist:")
    print(ds_hist)
   
    print("===== Check if T-grid of X (lon) are the same =====")
    
    geolon_t = ds_hist["geolon_t"].transpose("yt_ocean", "xt_ocean").to_numpy()
    x = ds_hgrid["x"].isel(nyp=slice(1, None, 2), nxp=slice(1, None, 2)).transpose("nyp", "nxp").to_numpy()
    print("History geolon_t: ", geolon_t)
    print("hgrid x[nyp=1::2, nxp=1::2]", x)


    x_exactly_same, x_all_close = test_array_equality(geolon_t, x, atol=atol, rtol=rtol)
    print("Are they exactly the same? ", x_exactly_same)
    print(f"Are they sufficiently close (atol={atol})? ", x_all_close)

    print("===== Check if T-grid of Y (lat) are the same =====")
    
    geolat_t = ds_hist["geolat_t"].transpose("yt_ocean", "xt_ocean").to_numpy()
    y = ds_hgrid["y"].isel(nyp=slice(1, None, 2), nxp=slice(1, None, 2)).transpose("nyp", "nxp").to_numpy()
    print("History geolat_t: ", geolat_t)
    print("hgrid y[nyp=1::2, nxp=1::2]", y)

    y_exactly_same, y_all_close = test_array_equality(geolat_t, y, atol=atol, rtol=rtol)
    print("Are they exactly the same? ", y_exactly_same)
    print(f"Are they sufficiently close (atol={atol})? ", y_all_close)
    
    print("===== Check if T-grid cell area can be computed =====")
    
    dx_mosaic = ds_hgrid["dx"].transpose("nyp", "nx").to_numpy()
    dy_mosaic = ds_hgrid["dy"].transpose("ny", "nxp").to_numpy()

    dx = dx_mosaic[1::2, 0::2] + dx_mosaic[1::2, 1::2]
    dy = dy_mosaic[0::2, 1::2] + dy_mosaic[1::2, 1::2]

    area_hgrid = dx*dy
    area_t = ds_hist["area_t"].transpose("yt_ocean", "xt_ocean").to_numpy()

    print(area_hgrid)
    print(area_t)

    area_exactly_same, area_all_close = test_array_equality(area_hgrid, area_t, rtol=rtol, atol=atol)
    print("Are area exactly the same? ", area_exactly_same)
    print("Are area sufficiently close? ", area_all_close)
   
    print("Verify that hgrid is consistent with history file? ", x_all_close and y_all_close and area_all_close) 

if __name__ == "__main__":

    ocean_hgrid_file = Path("data/ocean_hgrid.nc")
    ocean_history_file = Path("data/0017010101.ocean_year.nc")

    verify_history_and_hgrid_files(ocean_hgrid_file, ocean_history_file)
