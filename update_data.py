# =============================================================================
#  DATA UPDATER — Extend NIFTY 50 dataset to current date
#  Run this script locally on your machine to fetch live data
#  Requirements: pip install yfinance pandas
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# =============================================================================
# STEP 1: Download all data up to TODAY
# =============================================================================
TODAY = str(date.today())
print(f"Downloading data up to {TODAY}...")

# NIFTY 50
nifty = yf.download("^NSEI", start="2015-01-01", end=TODAY,
                    auto_adjust=True, progress=False)
nifty = nifty[["Close"]].copy()
nifty.columns = ["NIFTY_Close"]
nifty["Log_Return"] = np.log(nifty["NIFTY_Close"] / nifty["NIFTY_Close"].shift(1))

# India VIX (fear index)
vix = yf.download("^INDIAVIX", start="2015-01-01", end=TODAY,
                  auto_adjust=True, progress=False)[["Close"]]
vix.columns = ["VIX"]

# Crude Oil (WTI)
oil = yf.download("CL=F", start="2015-01-01", end=TODAY,
                  auto_adjust=True, progress=False)[["Close"]]
oil.columns = ["Crude_Oil"]

# USD/INR exchange rate
usdinr = yf.download("USDINR=X", start="2015-01-01", end=TODAY,
                     auto_adjust=True, progress=False)[["Close"]]
usdinr.columns = ["USDINR"]

print(f"NIFTY 50  : {len(nifty):,} rows")
print(f"VIX       : {len(vix):,} rows")
print(f"Crude Oil : {len(oil):,} rows")
print(f"USD/INR   : {len(usdinr):,} rows")

# =============================================================================
# STEP 2: Merge all on NIFTY trading dates
# =============================================================================
df = nifty.copy()
df = df.join(vix,    how="left")
df = df.join(oil,    how="left")
df = df.join(usdinr, how="left")

# Forward-fill missing values (holidays in other markets)
df = df.fillna(method="ffill")

# =============================================================================
# STEP 3: Add Macro Data (monthly, mapped to daily)
# NOTE: Update these values as new RBI/MOSPI data is released
# Source: https://dbie.rbi.org.in  |  https://mospi.gov.in
# =============================================================================

# Monthly Repo Rate history (RBI)
repo_schedule = {
    # (year, month): rate
    (2015, 1): 7.75,  (2015, 3): 7.50,  (2015, 6): 7.25,  (2015, 9): 6.75,
    (2016, 4): 6.50,  (2016, 10): 6.25,
    (2017, 8): 6.00,
    (2018, 6): 6.25,  (2018, 8): 6.50,
    (2019, 2): 6.25,  (2019, 4): 6.00,  (2019, 6): 5.75,
    (2019, 8): 5.40,  (2019, 10): 5.15,
    (2020, 3): 4.40,  (2020, 5): 4.00,
    (2022, 5): 4.40,  (2022, 6): 4.90,  (2022, 8): 5.40,
    (2022, 9): 5.90,  (2022, 12): 6.25,
    (2023, 2): 6.50,
    (2025, 2): 6.25,  (2025, 4): 6.00,   # ← Recent RBI cuts
}

def get_repo_rate(date):
    rate = 8.0  # default start
    for (y, m), r in sorted(repo_schedule.items()):
        if (date.year, date.month) >= (y, m):
            rate = r
    return rate

df["Repo_Rate"] = [get_repo_rate(d) for d in df.index]

# Monthly CPI (approximate — update from MOSPI)
cpi_annual = {
    2015: 5.5, 2016: 4.9, 2017: 3.6, 2018: 3.9, 2019: 4.8,
    2020: 6.2, 2021: 5.6, 2022: 6.7, 2023: 5.4, 2024: 4.6,
    2025: 4.0   # ← Estimate — update with actual MOSPI data
}
df["CPI"] = [cpi_annual.get(d.year, 5.0) for d in df.index]

# Drop rows with missing core data
df = df.dropna(subset=["NIFTY_Close", "Log_Return"])

# =============================================================================
# STEP 4: Save
# =============================================================================
df.to_csv("nifty_global_dataset.csv")
print(f"\n✅ Dataset saved: {len(df):,} rows")
print(f"   Period: {df.index[0].date()} → {df.index[-1].date()}")
print(f"\n   Columns: {list(df.columns)}")
print("\n   Now run: python nifty_direction_predictor.py")

# =============================================================================
# QUICK CHECK — Recent data (last 10 rows)
# =============================================================================
print("\n── Last 10 rows (most recent data) ──")
print(df.tail(10).to_string())
