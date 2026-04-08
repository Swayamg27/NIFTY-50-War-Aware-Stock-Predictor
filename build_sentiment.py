# =============================================================================
#  BUILD SENTIMENT — Rebuild geopolitical sentiment scores
#  Run this to regenerate sentiment data or add new events
#  Author: [Your Name]
# =============================================================================

import pandas as pd
import numpy as np

# Add new events here as they happen
# Format: ("YYYY-MM-DD", "YYYY-MM-DD", "headline description", sentiment_score)
# Sentiment: -1.0 (extreme crisis) to +1.0 (very positive)
WAR_EVENTS = [
    # 2015-2016
    ("2015-01-01", "2015-03-31", "Oil prices crash amid global supply glut", -0.45),
    ("2015-11-13", "2015-12-31", "Paris terror attacks shock global markets", -0.60),
    ("2016-06-24", "2016-07-10", "Brexit shock: pound collapses, markets plunge", -0.70),
    # 2017-2018
    ("2017-08-05", "2017-09-15", "North Korea missile tests rattle Asia markets", -0.55),
    ("2018-03-22", "2018-07-06", "US-China trade war begins; tariffs escalate", -0.50),
    ("2018-10-01", "2018-12-31", "US-China trade war deepens; Indian markets hit", -0.45),
    # 2019
    ("2019-05-05", "2019-06-30", "Trump escalates China tariffs; global growth fears", -0.55),
    ("2019-08-05", "2019-08-31", "US-China trade war: China weakens yuan; panic", -0.65),
    ("2019-09-14", "2019-10-15", "Saudi Aramco attack by Iran drones; oil +15%", -0.70),
    # 2020 COVID
    ("2020-01-20", "2020-02-15", "China coronavirus outbreak; markets uneasy", -0.40),
    ("2020-02-20", "2020-03-23", "COVID-19 pandemic declared; global markets crash", -0.90),
    ("2020-03-24", "2020-05-31", "Central banks stimulus; recovery hopes", 0.30),
    ("2020-06-01", "2020-12-31", "Vaccine hopes lift markets", 0.20),
    # 2021
    ("2021-01-06", "2021-01-20", "US Capitol storming shocks markets", -0.50),
    ("2021-09-01", "2021-10-31", "China Evergrande debt crisis; contagion fears", -0.55),
    ("2021-11-01", "2021-12-31", "Omicron variant triggers market selloff", -0.45),
    # 2022 Russia-Ukraine
    ("2022-01-20", "2022-02-23", "Russia masses troops on Ukraine border", -0.60),
    ("2022-02-24", "2022-03-31", "Russia invades Ukraine; oil hits $130", -0.85),
    ("2022-04-01", "2022-06-30", "Ukraine war; sanctions hit energy and food", -0.55),
    ("2022-07-01", "2022-09-30", "Inflation 40-year high; Fed hikes aggressively", -0.50),
    ("2022-10-01", "2022-12-31", "Ukraine war continues; energy crisis Europe", -0.45),
    # 2023
    ("2023-01-01", "2023-03-31", "SVB and Credit Suisse collapse shocks markets", -0.65),
    ("2023-04-01", "2023-09-30", "Markets stabilize; AI boom drives tech rally", 0.35),
    ("2023-10-07", "2023-11-30", "Hamas attacks Israel; Middle East war fears", -0.80),
    ("2023-12-01", "2023-12-31", "Houthi Red Sea attacks disrupt global shipping", -0.50),
    # 2024
    ("2024-01-01", "2024-03-31", "Iran strikes Israel; Hormuz oil supply fears", -0.75),
    ("2024-04-01", "2024-06-30", "Middle East eases slightly; India election", -0.25),
    ("2024-07-01", "2024-09-30", "Modi wins; markets rally on policy continuity", 0.40),
    ("2024-10-01", "2024-12-31", "Trump wins US election; tariff fears return", -0.35),

    # ── ADD NEW EVENTS BELOW ──────────────────────────────────────────────────
    # ("2025-04-01", "2025-04-30", "US tariff shocks: 145% on China, markets rout", -0.75),
    # ("2025-05-07", "2025-05-31", "India-Pakistan military conflict; NIFTY drops", -0.80),
    # ("2025-XX-XX", "2025-XX-XX", "Add your event here", -0.XX),
]

def build_sentiment(csv_path):
    df = pd.read_csv(csv_path)
    # Try to detect date column
    date_col = [c for c in df.columns if 'date' in c.lower() or 'Date' in c][0]
    dates = pd.to_datetime(df[date_col])
    n = len(dates)

    np.random.seed(42)
    sentiment = np.ones(n) * 0.05  # slight positive baseline

    for start, end, _, base_sent in WAR_EVENTS:
        mask = (dates >= start) & (dates <= end)
        if mask.sum() > 0:
            n_days = mask.sum()
            decay  = np.linspace(1.0, 0.4, n_days)
            noise  = np.random.normal(0, 0.08, n_days)
            sentiment[mask.values] = base_sent * decay + noise

    sentiment = np.clip(sentiment, -1, 1)
    sent_series = pd.Series(sentiment, index=dates.values)

    df["Sentiment_Score"]  = np.round(sentiment, 4)
    df["Sentiment_MA3"]    = np.round(sent_series.rolling(3).mean().values, 4)
    df["Sentiment_MA7"]    = np.round(sent_series.rolling(7).mean().values, 4)
    df["Sentiment_Std7"]   = np.round(sent_series.rolling(7).std().values, 4)
    df["Sentiment_Change"] = np.round(sent_series.diff().values, 4)
    df["War_Crisis_Flag"]  = (sentiment < -0.5).astype(int)

    out = csv_path.replace(".csv", "_with_sentiment.csv")
    df.to_csv(out, index=False)
    print(f"✅ Saved: {out}")
    print(f"   War/crisis days: {df['War_Crisis_Flag'].sum()} ({df['War_Crisis_Flag'].mean()*100:.1f}%)")
    return df

if __name__ == "__main__":
    build_sentiment("nifty_global_sentiment_dataset.csv")
    print("\nTo add new events, edit the WAR_EVENTS list at the top of this file.")
