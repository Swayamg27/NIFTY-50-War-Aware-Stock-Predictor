# =============================================================================
#  NIFTY 50 — Stock Price Direction Predictor (War-Aware Edition)
#  Technical + Global Macro + Geopolitical Sentiment Features
#  Algorithm : Random Forest Classifier
#  Author    : [Your Name]
#  Data      : NIFTY 50 (2015–2024) | VIX | Crude Oil | USD/INR |
#              CPI | Repo Rate | Geopolitical Sentiment Scores
# =============================================================================
#
#  GEOPOLITICAL EVENTS COVERED:
#  ✅ COVID-19 Crash (2020)
#  ✅ Russia-Ukraine War (Feb 2022)
#  ✅ Israel-Gaza Conflict (Oct 2023)
#  ✅ Iran-Israel Direct Strike (Apr 2024)
#  ✅ US-China Trade War (2018-2024)
#  ✅ Houthi Red Sea Shipping Attacks (2023-2024)
#  ✅ Saudi Aramco Drone Strike (Sep 2019)
#  ✅ US-Iran Tensions / Strait of Hormuz Risk
#  ✅ India-Pakistan Tensions
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#c9d1d9", "grid.color": "#21262d",
    "axes.spines.top": False, "axes.spines.right": False,
})
CYAN="#58a6ff"; GREEN="#3fb950"; RED="#f85149"; YELLOW="#e3b341"
PURPLE="#bc8cff"; ORANGE="#f0883e"; GREY="#8b949e"; BG="#0d1117"

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("nifty_global_sentiment_dataset.csv", parse_dates=["Date"])
df = df.set_index("Date").sort_index()

print("=" * 70)
print("  NIFTY 50 WAR-AWARE DIRECTION PREDICTOR")
print("=" * 70)
print(f"  Period       : {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Observations : {len(df):,} trading days")
print(f"\n  Geopolitical Events in Dataset:")
for e in ["COVID-19 Crash (2020)", "Russia-Ukraine War (Feb 2022)",
          "Israel-Gaza / Iran-Israel (2023-2024)", "US-China Trade War (2018-2024)",
          "Houthi Red Sea Attacks (2023-2024)", "Saudi Aramco Attack (Sep 2019)",
          "US-Iran / Strait of Hormuz Tensions"]:
    print(f"    ✅ {e}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

# Technical Indicators
df["Return_1d"]  = df["Log_Return"]
df["Return_5d"]  = df["NIFTY_Close"].pct_change(5)
df["Return_20d"] = df["NIFTY_Close"].pct_change(20)
df["MA_5"]  = df["NIFTY_Close"].rolling(5).mean()
df["MA_20"] = df["NIFTY_Close"].rolling(20).mean()
df["MA_50"] = df["NIFTY_Close"].rolling(50).mean()
df["MA_ratio_5_20"]  = df["MA_5"]  / df["MA_20"]
df["MA_ratio_20_50"] = df["MA_20"] / df["MA_50"]
df["Vol_5d"]  = df["Log_Return"].rolling(5).std()
df["Vol_20d"] = df["Log_Return"].rolling(20).std()

delta = df["NIFTY_Close"].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["RSI_14"] = 100 - (100 / (1 + gain / loss))

ema12 = df["NIFTY_Close"].ewm(span=12, adjust=False).mean()
ema26 = df["NIFTY_Close"].ewm(span=26, adjust=False).mean()
df["MACD"]        = ema12 - ema26
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

bb_mid = df["NIFTY_Close"].rolling(20).mean()
bb_std = df["NIFTY_Close"].rolling(20).std()
df["BB_position"] = (df["NIFTY_Close"] - (bb_mid - 2*bb_std)) / (4*bb_std)

# Global Market Features
df["VIX_change"]    = df["VIX"].pct_change()
df["VIX_MA5"]       = df["VIX"].rolling(5).mean()
df["Oil_change"]    = df["Crude_Oil"].pct_change()
df["Oil_MA5"]       = df["Crude_Oil"].rolling(5).mean()
df["USDINR_change"] = df["USDINR"].pct_change()
df["High_Stress"]   = (df["VIX"] > 25).astype(int)

# Geopolitical Sentiment Features
df["Sent_Score"]      = df["Sentiment_Score"]
df["Sent_MA3"]        = df["Sentiment_MA3"]
df["Sent_MA7"]        = df["Sentiment_MA7"]
df["Sent_Change"]     = df["Sentiment_Change"]
df["War_Flag"]        = df["War_Crisis_Flag"]
df["Sent_Stress"]     = (df["Sentiment_Score"] < -0.4).astype(int)
df["VIX_x_Sentiment"] = df["VIX"] * (-df["Sent_Score"])
df["Oil_War_Signal"]  = df["Oil_change"].abs() * df["War_Flag"]

# Target
df["Target"] = (df["NIFTY_Close"].shift(-1) > df["NIFTY_Close"]).astype(int)
df.dropna(inplace=True)

up_pct = df["Target"].mean() * 100
print(f"\n  After engineering : {len(df):,} rows, {df.shape[1]} features")
print(f"  UP days   : {df['Target'].sum():,} ({up_pct:.1f}%)")
print(f"  DOWN days : {(df['Target']==0).sum():,} ({100-up_pct:.1f}%)")
print(f"  War/crisis days : {df['War_Flag'].sum():,} ({df['War_Flag'].mean()*100:.1f}%)")

# =============================================================================
# 3. FEATURES + SPLIT
# =============================================================================
FEATURES = [
    "Return_1d", "Return_5d", "Return_20d",
    "MA_ratio_5_20", "MA_ratio_20_50",
    "Vol_5d", "Vol_20d", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist", "BB_position",
    "VIX", "VIX_change", "VIX_MA5", "High_Stress",
    "Oil_change", "Oil_MA5", "USDINR_change",
    "CPI", "Repo_Rate",
    "Sent_Score", "Sent_MA3", "Sent_MA7",
    "Sent_Change", "War_Flag", "Sent_Stress",
    "VIX_x_Sentiment", "Oil_War_Signal",
]

X = df[FEATURES]
y = df["Target"]

split = "2023-01-01"
X_train, X_test = X[X.index < split], X[X.index >= split]
y_train, y_test = y[y.index < split], y[y.index >= split]

print(f"\n  Train : {len(X_train):,} days (2015-2022)")
print(f"  Test  : {len(X_test):,}  days (2023-2024)")

# Scale
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# 4. TRAIN + EVALUATE
# =============================================================================
model = RandomForestClassifier(
    n_estimators=500, max_depth=8,
    min_samples_split=20, min_samples_leaf=10,
    random_state=42, class_weight="balanced", n_jobs=-1,
)
model.fit(X_train_sc, y_train)

y_pred      = model.predict(X_test_sc)
y_pred_prob = model.predict_proba(X_test_sc)[:, 1]
cv_scores   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc     = auc(fpr, tpr)

test_df = pd.DataFrame({
    "prob_up":     y_pred_prob,
    "actual":      y_test.values,
    "pred":        y_pred,
    "correct":     (y_pred == y_test.values).astype(int),
    "war_flag":    df.loc[y_test.index, "War_Flag"].values,
    "high_stress": df.loc[y_test.index, "High_Stress"].values,
    "sentiment":   df.loc[y_test.index, "Sent_Score"].values,
}, index=y_test.index)

war_acc    = test_df[test_df["war_flag"]==1]["correct"].mean() * 100
stress_acc = test_df[test_df["high_stress"]==1]["correct"].mean() * 100
normal_acc = test_df[(test_df["war_flag"]==0) & (test_df["high_stress"]==0)]["correct"].mean() * 100
overall    = test_df["correct"].mean() * 100

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"  CV Accuracy (5-fold)  : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"  Test Accuracy         : {overall:.2f}%")
print(f"  ROC-AUC               : {roc_auc:.4f}")
print(f"\n  By Market Regime:")
print(f"    Normal market        : {normal_acc:.1f}%")
print(f"    High Stress (VIX>25) : {stress_acc:.1f}%")
print(f"    Active War/Crisis    : {war_acc:.1f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["DOWN","UP"]))

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("  Top 10 Features:")
for i, (f, v) in enumerate(importances.head(10).items(), 1):
    print(f"    {i:2}. {f:<26} {v:.4f}")

# =============================================================================
# 5. CHARTS
# =============================================================================
TECH = ["Return_1d","Return_5d","Return_20d","MA_ratio_5_20","MA_ratio_20_50",
        "Vol_5d","Vol_20d","RSI_14","MACD","MACD_Signal","MACD_Hist","BB_position"]
GEO  = ["Sent_Score","Sent_MA3","Sent_MA7","Sent_Change","War_Flag",
        "Sent_Stress","VIX_x_Sentiment","Oil_War_Signal"]
GLOB = ["VIX","VIX_change","VIX_MA5","High_Stress","Oil_change","Oil_MA5","USDINR_change"]
MAC  = ["CPI","Repo_Rate"]

def fcolor(f):
    if f in GEO:  return ORANGE
    if f in GLOB: return RED
    if f in MAC:  return PURPLE
    return CYAN

WAR_MARKERS = [
    ("2020-02-20","COVID\nCrash", RED),
    ("2022-02-24","Russia-\nUkraine", ORANGE),
    ("2023-10-07","Israel-\nGaza", YELLOW),
    ("2024-01-15","Iran-\nIsrael", RED),
    ("2018-03-22","US-China\nTrade War", PURPLE),
    ("2019-09-14","Aramco\nAttack", ORANGE),
]

# Chart 1: Feature Importance
fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
ax.set_facecolor(BG)
top20 = importances.head(20)
ax.barh(top20.index[::-1], top20.values[::-1],
        color=[fcolor(f) for f in top20.index[::-1]], height=0.65, edgecolor="none")
ax.set_title("Feature Importance — War-Aware NIFTY Direction Predictor",
             fontsize=13, fontweight="bold", color="#e6edf3", pad=14)
ax.set_xlabel("Importance Score", fontsize=10)
ax.legend(handles=[
    mpatches.Patch(color=ORANGE, label="Geopolitical Sentiment"),
    mpatches.Patch(color=RED,    label="Global Market (VIX/Oil/FX)"),
    mpatches.Patch(color=PURPLE, label="Macro (CPI/Repo Rate)"),
    mpatches.Patch(color=CYAN,   label="Technical Indicators"),
], fontsize=9, framealpha=0, loc="lower right")
plt.tight_layout()
plt.savefig("chart1_feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅  chart1_feature_importance.png")

# Chart 2: Sentiment + War Timeline
sent = df["Sent_Score"]
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=BG, sharex=True)
fig.suptitle("NIFTY 50 — Geopolitical Sentiment vs Market (2015-2024)",
             fontsize=14, fontweight="bold", color="#e6edf3")

axes[0].plot(df.index, df["NIFTY_Close"], color=CYAN, lw=1.2)
axes[0].set_facecolor(BG); axes[0].set_ylabel("NIFTY 50", fontsize=9)
axes[0].set_title("Index Level", fontsize=10, color=GREY)

axes[1].fill_between(df.index, sent, 0, where=sent>=0, alpha=0.5, color=GREEN, label="Positive")
axes[1].fill_between(df.index, sent, 0, where=sent<0,  alpha=0.5, color=RED,   label="Negative")
axes[1].plot(df.index, df["Sent_MA7"], color=YELLOW, lw=1.5, label="7-day avg")
axes[1].axhline(-0.5, color=ORANGE, lw=0.8, ls="--", alpha=0.7, label="War threshold")
axes[1].set_facecolor(BG); axes[1].set_ylabel("Sentiment", fontsize=9)
axes[1].set_title("Geopolitical Sentiment Score (-1=Crisis, +1=Positive)", fontsize=10, color=GREY)
axes[1].legend(fontsize=8, framealpha=0, loc="lower left", ncol=4)

axes[2].fill_between(df.index, df["VIX"], alpha=0.3, color=PURPLE)
axes[2].plot(df.index, df["VIX"], color=PURPLE, lw=1.2)
axes[2].axhline(25, color=RED, lw=0.8, ls="--", alpha=0.7, label="Stress threshold (25)")
axes[2].set_facecolor(BG); axes[2].set_ylabel("India VIX", fontsize=9)
axes[2].set_xlabel("Date", fontsize=9); axes[2].legend(fontsize=8, framealpha=0)
axes[2].set_title("India VIX — Fear Index", fontsize=10, color=GREY)

for ds, label, color in WAR_MARKERS:
    d = pd.Timestamp(ds)
    for ax in axes:
        ax.axvline(d, color=color, lw=1, alpha=0.55, ls="--")
    axes[0].text(d, df["NIFTY_Close"].max()*0.92, label,
                fontsize=6.5, color=color, ha="center", rotation=90, va="top")

for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("chart2_sentiment_timeline.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  chart2_sentiment_timeline.png")

# Chart 3: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
ax.set_facecolor(BG)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["DOWN","UP"]).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix (Test 2023-2024)", fontsize=12,
             fontweight="bold", color="#e6edf3", pad=12)
for t in ax.texts: t.set_color("#e6edf3"); t.set_fontsize(14)
plt.tight_layout()
plt.savefig("chart3_confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  chart3_confusion_matrix.png")

# Chart 4: ROC Curve
fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(fpr, tpr, color=CYAN, lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], color=GREY, lw=1, ls="--", label="Random Guess")
ax.fill_between(fpr, tpr, alpha=0.12, color=CYAN)
ax.set_title("ROC Curve — War-Aware NIFTY Predictor", fontsize=12,
             fontweight="bold", color="#e6edf3", pad=12)
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.legend(fontsize=10, framealpha=0)
plt.tight_layout()
plt.savefig("chart4_roc_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  chart4_roc_curve.png")

# Chart 5: Regime Accuracy
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
vals = [normal_acc,
        stress_acc if not np.isnan(stress_acc) else 0,
        war_acc    if not np.isnan(war_acc)    else 0,
        overall]
bars = ax.bar(["Normal\nMarket","High Stress\n(VIX>25)","Active War/\nCrisis","Overall"],
              vals, color=[GREEN,YELLOW,RED,CYAN], width=0.5, edgecolor="none")
ax.axhline(50, color=GREY, lw=1.2, ls="--", label="Random baseline (50%)")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color="#e6edf3")
ax.set_ylim(0, 80)
ax.set_title("Accuracy by Market Regime — Normal vs War/Crisis",
             fontsize=12, fontweight="bold", color="#e6edf3", pad=12)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.legend(fontsize=9, framealpha=0)
plt.tight_layout()
plt.savefig("chart5_regime_accuracy.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  chart5_regime_accuracy.png")

# Chart 6: Prediction Timeline
price_test = df.loc[y_test.index, "NIFTY_Close"]
sent_test  = df.loc[y_test.index, "Sent_Score"]
fig, axes  = plt.subplots(3, 1, figsize=(14, 9), facecolor=BG, sharex=True)
fig.suptitle("Model Predictions vs Reality (2023-2024)", fontsize=13,
             fontweight="bold", color="#e6edf3")

axes[0].plot(price_test.index, price_test.values, color=CYAN, lw=1.5)
axes[0].set_facecolor(BG); axes[0].set_ylabel("NIFTY Close", fontsize=9)
axes[0].set_title("NIFTY 50 Price", fontsize=10, color=GREY)

axes[1].fill_between(test_df.index, sent_test, 0, where=sent_test>=0, alpha=0.5, color=GREEN)
axes[1].fill_between(test_df.index, sent_test, 0, where=sent_test<0,  alpha=0.5, color=RED)
axes[1].set_facecolor(BG); axes[1].set_ylabel("Sentiment", fontsize=9)
axes[1].set_title("Geopolitical Sentiment", fontsize=10, color=GREY)

axes[2].plot(test_df.index, test_df["prob_up"], color=YELLOW, lw=1.2, label="P(UP tomorrow)")
axes[2].axhline(0.5, color=GREY, lw=0.8, ls="--")
axes[2].fill_between(test_df.index, 0.5, test_df["prob_up"],
                     where=test_df["prob_up"]>0.5, alpha=0.3, color=GREEN)
axes[2].fill_between(test_df.index, 0.5, test_df["prob_up"],
                     where=test_df["prob_up"]<=0.5, alpha=0.3, color=RED)
axes[2].set_facecolor(BG); axes[2].set_xlabel("Date", fontsize=9)
axes[2].set_ylabel("Probability", fontsize=9); axes[2].legend(fontsize=9, framealpha=0)
axes[2].set_title("Model Confidence", fontsize=10, color=GREY)

for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("chart6_prediction_timeline.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  chart6_prediction_timeline.png")

# Chart 7: CV Scores
fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
ax.set_facecolor(BG)
ax.bar(range(1,6), cv_scores*100, color=PURPLE, width=0.5, edgecolor="none")
ax.axhline(cv_scores.mean()*100, color=YELLOW, lw=1.5, ls="--",
           label=f"Mean = {cv_scores.mean()*100:.1f}%")
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=12,
             fontweight="bold", color="#e6edf3", pad=12)
ax.set_xlabel("Fold", fontsize=10); ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(range(1,6)); ax.set_ylim(40, 80); ax.legend(fontsize=9, framealpha=0)
plt.tight_layout()
plt.savefig("chart7_cv_scores.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  chart7_cv_scores.png")

print("\n" + "=" * 70)
print(f"  FINAL: Acc={overall:.1f}% | AUC={roc_auc:.3f} | War Acc={war_acc:.1f}%")
print("  All 7 charts saved. Ready for GitHub!")
print("=" * 70)
