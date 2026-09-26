import xarray as xr
import pandas as pd
import numpy as np
from pyproj import Transformer

# --------------------------------------------------
# 1. Find NSIDC NetCDF file
# --------------------------------------------------

file_path = "data/raw/nsidc/sic_pss25_20250401_am2_v06r00.nc"

print("Opening NSIDC file...")
ds = xr.open_dataset(file_path)

# --------------------------------------------------
# 2. Read sea-ice concentration
# --------------------------------------------------

ice = ds["cdr_seaice_conc"].isel(time=0)

x = ds["x"].values
y = ds["y"].values

print("Grid size:", ice.shape)

# --------------------------------------------------
# 3. Create X/Y grid
# --------------------------------------------------

X, Y = np.meshgrid(x, y)

ice_values = ice.values

# --------------------------------------------------
# 4. Convert EPSG:3412 → Latitude/Longitude
# --------------------------------------------------

print("Converting coordinates...")

transformer = Transformer.from_crs(
    "EPSG:3412",
    "EPSG:4326",
    always_xy=True
)

longitude, latitude = transformer.transform(X, Y)

# --------------------------------------------------
# 5. Flatten arrays
# --------------------------------------------------

latitude = latitude.flatten()
longitude = longitude.flatten()
ice_values = ice_values.flatten()

# --------------------------------------------------
# 6. Remove invalid values
# --------------------------------------------------

valid = (
    np.isfinite(latitude)
    & np.isfinite(longitude)
    & np.isfinite(ice_values)
)

latitude = latitude[valid]
longitude = longitude[valid]
ice_values = ice_values[valid]

# --------------------------------------------------
# 7. Keep Antarctic region
# --------------------------------------------------

antarctic = latitude <= -55

latitude = latitude[antarctic]
longitude = longitude[antarctic]
ice_values = ice_values[antarctic]

# --------------------------------------------------
# 8. Normalize concentration
# --------------------------------------------------

# NSIDC concentration may be stored as fraction (0–1)
# or percentage depending on product representation.

if np.nanmax(ice_values) <= 1.5:
    ice_values = ice_values * 100

ice_values = np.clip(ice_values, 0, 100)

# --------------------------------------------------
# 9. Create dataframe
# --------------------------------------------------

df = pd.DataFrame({
    "latitude": latitude,
    "longitude": longitude,
    "sea_ice_concentration": ice_values
})

# Add observation date
df["date"] = "2025-04-01"

# --------------------------------------------------
# 10. Save
# --------------------------------------------------

output = "data/antarctic_ice.csv"

df.to_csv(output, index=False)

print("\n================================")
print("REAL NSIDC DATA PROCESSED!")
print("================================")

print("Output:", output)
print("Records:", len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nSea-ice concentration:")
print(df["sea_ice_concentration"].describe())

print("\nLatitude range:")
print(df["latitude"].min(), "to", df["latitude"].max())

print("\nLongitude range:")
print(df["longitude"].min(), "to", df["longitude"].max())