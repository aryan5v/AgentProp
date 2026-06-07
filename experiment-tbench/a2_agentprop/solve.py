# A2 AGENTPROP-GUIDED. Following the AgentProp brief + extra-instructions:
#   "For numerical/data tasks, confirm units, schema, and formats BEFORE fitting."
#   Verifier node (tester) must INTERCEPT contradicted assumptions before finalize.
import sys
import pandas as pd

high = pd.read_csv("/app/daily_temp_sf_high.csv")
low = pd.read_csv("/app/daily_temp_sf_low.csv")

# --- PLANNER/CODER: inspect schemas & date formats before computing ---
print("HIGH date samples:", high["date"].tolist())
print("LOW  date samples:", low["date"].tolist())

# --- IMPLEMENTER: dates are heterogeneous in LOW (mixed '/' and '-', with time).
# Normalize separators, parse with explicit MDY format, then drop the time. ---
low["date"] = low["date"].astype(str).str.replace("/", "-", regex=False)
low["date"] = pd.to_datetime(low["date"], format="%m-%d-%Y %H:%M:%S").dt.normalize()
high["date"] = pd.to_datetime(high["date"], format="%Y-%m-%d")

high = high.rename(columns={"temperature": "temp_high"})
low = low.rename(columns={"temperature": "temp_low"})
merged = high.merge(low, how="inner", on="date")

# --- TESTER/VERIFIER INTERCEPT: contradicted-assumption guard ---
if len(merged) != len(high) or merged.isnull().any().any():
    print(f"VERIFIER INTERCEPT: merge produced {len(merged)} rows (expected {len(high)}); aborting.")
    sys.exit(1)

merged["difference"] = merged["temp_high"] - merged["temp_low"]
avg_diff = merged["difference"].mean()
print("merged rows:", len(merged), "avg_diff:", avg_diff)

# --- FINALIZER ---
with open("/app/avg_temp.txt", "w") as f:
    f.write(str(avg_diff))
