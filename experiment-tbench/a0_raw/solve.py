# A0 RAW: natural "merge on date" approach, no schema/format inspection first.
# This mirrors the reference solution's STRUCTURE (merge on date) but skips the
# heterogeneous-date-format normalization step a hasty agent would not anticipate.
import pandas as pd

high = pd.read_csv("/app/daily_temp_sf_high.csv")
low = pd.read_csv("/app/daily_temp_sf_low.csv")

merged = high.merge(low, on="date", suffixes=("_high", "_low"))
avg_diff = (merged["temperature_high"] - merged["temperature_low"]).mean()

with open("/app/avg_temp.txt", "w") as f:
    f.write(str(avg_diff))
print("merged rows:", len(merged), "avg_diff:", avg_diff)
