import xarray as xr
import os

files = [
    "data/raw/era5_extracted/data_stream-oper_stepType-instant.nc",
    "data/raw/era5_extracted/data_stream-wave_stepType-instant.nc"
]

for file_path in files:

    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(file_path))
    print("=" * 70)

    ds = xr.open_dataset(file_path, engine="netcdf4")

    print("\nDATASET:")
    print(ds)

    print("\nVARIABLES:")
    for name in ds.data_vars:
        print(" -", name)

    print("\nCOORDINATES:")
    for name in ds.coords:
        print(" -", name)

    ds.close()