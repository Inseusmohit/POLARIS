import xarray as xr
import pandas as pd
import numpy as np

oper_file = "data/raw/era5_extracted/data_stream-oper_stepType-instant.nc"
wave_file = "data/raw/era5_extracted/data_stream-wave_stepType-instant.nc"

print("Loading ERA5 operational data...")
oper = xr.open_dataset(oper_file)

print("Loading ERA5 wave data...")
wave = xr.open_dataset(wave_file)

# --------------------------------------------------
# Select 12:00 UTC
# --------------------------------------------------

oper = oper.sel(valid_time="2025-04-01T12:00:00")
wave = wave.sel(valid_time="2025-04-01T12:00:00")

# --------------------------------------------------
# Calculate wind speed
# --------------------------------------------------

wind_speed = np.sqrt(
    oper["u10"] ** 2 +
    oper["v10"] ** 2
)

# --------------------------------------------------
# Convert temperature Kelvin → Celsius
# --------------------------------------------------

temperature = oper["t2m"] - 273.15

# --------------------------------------------------
# Convert ERA5 data to DataFrames
# --------------------------------------------------

wind_df = wind_speed.to_dataframe(
    name="wind_speed"
).reset_index()

temp_df = temperature.to_dataframe(
    name="temperature"
).reset_index()

wave_df = wave["swh"].to_dataframe(
    name="wave_height"
).reset_index()

# --------------------------------------------------
# Merge environmental data
# --------------------------------------------------

df = wind_df.merge(
    temp_df,
    on=["latitude", "longitude"]
)

df = df.merge(
    wave_df,
    on=["latitude", "longitude"],
    how="left"
)

# --------------------------------------------------
# Remove missing values
# --------------------------------------------------

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "wind_speed",
        "temperature"
    ]
)

# --------------------------------------------------
# Add date/time
# --------------------------------------------------

df["date"] = "2025-04-01"
df["time"] = "12:00"

# --------------------------------------------------
# Save
# --------------------------------------------------

output = "data/era5_environment.csv"

df.to_csv(
    output,
    index=False
)

print()
print("=" * 50)
print("ERA5 ENVIRONMENT DATA CREATED")
print("=" * 50)

print("Output:", output)
print("Records:", len(df))

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Wind speed:")
print(df["wind_speed"].describe())

print()
print("Temperature °C:")
print(df["temperature"].describe())

print()
print("Wave height:")
print(df["wave_height"].describe())

oper.close()
wave.close()
