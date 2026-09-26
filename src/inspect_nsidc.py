import xarray as xr
import glob
import os

files = glob.glob("data/raw/nsidc/*.nc")

if not files:
    print("❌ No NetCDF file found.")
    print("Put a .nc file inside data/raw/nsidc/")
    exit()

file = files[0]

print("\n📂 File:")
print(file)

ds = xr.open_dataset(file)

print("\n========== DATASET ==========")
print(ds)

print("\n========== VARIABLES ==========")

for name in ds.data_vars:
    print(name)

print("\n========== COORDINATES ==========")

for name in ds.coords:
    print(name)

print("\n========== ATTRIBUTES ==========")

for key, value in ds.attrs.items():
    print(key, ":", value)