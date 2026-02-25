"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         RETAIL DEMAND FORECASTING — END-TO-END PIPELINE SHOWCASE           ║
║                                                                              ║
║  Author  : David Bravo · https://bravoaidatastudio.com                      ║
║  Project : Demand Forecasting & ELT for a Bolivian Home Improvement Retailer║
║  Stack   : Python · XGBoost · dbt · Snowflake · Airbyte                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script walks through the complete data science pipeline step by step:

    STEP 1 │ Data ingestion simulation & quality audit
    STEP 2 │ Feature engineering (calendar, lags, rolling, Bolivian holidays)
    STEP 3 │ Exploratory analysis — key statistical insights
    STEP 4 │ Model training with Time Series Cross-Validation
    STEP 5 │ 30-day demand forecast generation
    STEP 6 │ Model explainability — feature importance
    STEP 7 │ Business impact summary

Run this script to reproduce the full pipeline on the synthetic dataset:
    $ python src/generate_synthetic_data.py   # generate data first
    $ python pipeline_showcase.py              # then run this

For the complete case study and portfolio:
    https://bravoaidatastudio.com/portfolio/
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH    = PROJECT_ROOT / "data" / "raw_sales.csv"
OUTPUT_DIR   = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str, width: int = 72) -> None:
    """Print a formatted section banner to stdout."""
    bar = "─" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def tick(label: str) -> None:
    """Print a checkmark step."""
    print(f"  ✔  {label}")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


PALETTE = ["#2E86AB", "#E84855", "#F4A261", "#2A9D8F", "#8338EC", "#06D6A0", "#FB5607", "#3A86FF"]

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
})

BOLIVIAN_HOLIDAYS = {
    (1, 1): "Año Nuevo",
    (3, 23): "Día del Mar",
    (5, 1): "Día del Trabajo",
    (6, 21): "Año Nuevo Aymara",
    (8, 6): "Independencia",
    (11, 2): "Día de los Difuntos",
    (12, 25): "Navidad",
}

STORE_LABELS = {
    "ST-001": "La Paz",
    "ST-002": "Santa Cruz",
    "ST-003": "Cochabamba",
    "ST-004": "El Alto",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA INGESTION & QUALITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def step1_load_and_audit() -> pd.DataFrame:
    banner("STEP 1 │ DATA INGESTION & QUALITY AUDIT")

    if not DATA_PATH.exists():
        print(f"\n  ⚠  Data file not found at {DATA_PATH}")
        print("     Run:  python src/generate_synthetic_data.py\n")
        sys.exit(1)

    t0 = time.time()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    elapsed = time.time() - t0

    df = df.sort_values("date").reset_index(drop=True)
    df["store_label"] = df["store_id"].map(STORE_LABELS)

    tick(f"Loaded {len(df):,} rows in {elapsed:.2f}s — {DATA_PATH.name}")
    tick(f"Date range  : {df.date.min().date()} → {df.date.max().date()}")
    tick(f"SKUs        : {df.sku_id.nunique()} unique")
    tick(f"Stores      : {df.store_id.nunique()} (La Paz, Santa Cruz, Cochabamba, El Alto)")
    tick(f"Categories  : {df.category.nunique()}")

    # Quality checks
    missing = df.isnull().sum().sum()
    neg_qty = (df.quantity_sold < 0).sum()
    zero_pct = (df.quantity_sold == 0).mean() * 100

    print(f"\n  Quality audit:")
    tick(f"Missing values  : {missing}")
    tick(f"Negative qty    : {neg_qty}")
    tick(f"Zero-sale days  : {zero_pct:.1f}%  (stockouts / closed days)")

    print(f"\n  Quantity sold — descriptive statistics:")
    print(df["quantity_sold"].describe().rename("qty_sold").to_frame().T.to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def step2_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    banner("STEP 2 │ FEATURE ENGINEERING")

    df = df.copy()

    # ── Calendar features ────────────────────────────────────────────────────
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["year"]         = df["date"].dt.year
    df["year_frac"]    = (df["date"] - df["date"].min()).dt.days / 365.25  # trend proxy
    tick("Calendar features created (day, week, month, quarter, year trend)")

    # ── Cyclical encoding — preserves periodicity for tree models ────────────
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["woy_sin"]  = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"]  = np.cos(2 * np.pi * df["week_of_year"] / 52)
    tick("Cyclical sin/cos encoding applied (day, month, week)")

    # ── Bolivian domain features ─────────────────────────────────────────────
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 5).astype(int)
    df["is_month_end"]   = (df["day_of_month"] >= 25).astype(int)
    df["is_rainy_season"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["is_holiday"]     = df.apply(
        lambda r: 1 if (r["month"], r["day_of_month"]) in BOLIVIAN_HOLIDAYS else 0, axis=1
    )
    tick("Bolivian domain features: rainy season, holidays, weekend flags")

    # ── Lag features — per SKU×Store group ───────────────────────────────────
    group_cols = ["sku_id", "store_id"]
    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = df.groupby(group_cols)["quantity_sold"].shift(lag)
    tick("Lag features created: t−1, t−7, t−14, t−21, t−28")

    # ── Rolling statistics ────────────────────────────────────────────────────
    for window in [7, 14, 30]:
        base = df.groupby(group_cols)["quantity_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1)
        )
        df[f"roll_mean_{window}"] = df.groupby(group_cols)["quantity_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}"]  = df.groupby(group_cols)["quantity_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
        df[f"roll_max_{window}"]  = df.groupby(group_cols)["quantity_sold"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
    tick("Rolling statistics created: mean/std/max over 7, 14, 30-day windows")

    # ── Categorical encoding ──────────────────────────────────────────────────
    df["category_code"] = df["category"].astype("category").cat.codes
    df["store_code"]    = df["store_id"].astype("category").cat.codes
    tick("Categorical encoding: category, store_id")

    # ── Remove NaN rows generated by lags ────────────────────────────────────
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    tick(f"Dropped {before - len(df):,} rows with NaN (lag warm-up period)")

    # ── Define feature columns ────────────────────────────────────────────────
    exclude = {"date", "quantity_sold", "sku_id", "store_id", "category",
               "store_label", "stock_level"}
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ["int64", "float64", "int32", "float32"]]

    print(f"\n  Total features ready for modeling: {len(feature_cols)}")
    print(f"  {feature_cols[:10]} ... (+{len(feature_cols)-10} more)")

    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EXPLORATORY ANALYSIS (STATISTICAL INSIGHTS)
# ─────────────────────────────────────────────────────────────────────────────

def step3_eda_insights(df: pd.DataFrame) -> None:
    banner("STEP 3 │ EXPLORATORY ANALYSIS — KEY INSIGHTS")

    # Day-of-week effect
    dow_avg = df.groupby("day_of_week")["quantity_sold"].mean()
    sat_lift = (dow_avg[5] / dow_avg[[0,1,2,3]].mean() - 1) * 100
    print(f"  Day-of-week lift — Saturday vs Mon–Thu: +{sat_lift:.1f}%")

    # Seasonal effect per category
    season_pivot = df.groupby(["category","is_rainy_season"])["quantity_sold"].mean().unstack()
    if 1 in season_pivot.columns and 0 in season_pivot.columns:
        season_pivot["lift_pct"] = (season_pivot[1] / season_pivot[0] - 1) * 100
        top_rainy = season_pivot["lift_pct"].idxmax()
        top_dry   = season_pivot["lift_pct"].idxmin()
        print(f"  Top rainy-season category   : {top_rainy} ({season_pivot.loc[top_rainy,'lift_pct']:+.1f}%)")
        print(f"  Top dry-season category     : {top_dry} ({season_pivot.loc[top_dry,'lift_pct']:+.1f}%)")

    # Holiday effect
    holiday_avg = df[df["is_holiday"] == 1]["quantity_sold"].mean()
    normal_avg  = df[df["is_holiday"] == 0]["quantity_sold"].mean()
    print(f"  Holiday vs normal day sales : {holiday_avg:.1f} vs {normal_avg:.1f} ({(holiday_avg/normal_avg-1)*100:+.1f}%)")

    # YoY growth
    yoy = df.groupby("year")["quantity_sold"].sum()
    if 2024 in yoy and 2023 in yoy:
        print(f"  YoY growth 2023→2024        : +{(yoy[2024]/yoy[2023]-1)*100:.1f}%")
    if 2025 in yoy and 2024 in yoy:
        print(f"  YoY growth 2024→2025        : +{(yoy[2025]/yoy[2024]-1)*100:.1f}%")

    # Store share
    store_share = df.groupby("store_label")["quantity_sold"].sum()
    store_share_pct = (store_share / store_share.sum() * 100).round(1)
    print(f"\n  Store sales share:")
    for store, pct in store_share_pct.sort_values(ascending=False).items():
        bar = "█" * int(pct / 2)
        print(f"    {store:<15} {bar} {pct:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL TRAINING WITH TIME SERIES CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def step4_train_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    banner("STEP 4 │ MODEL TRAINING — XGBoost + TimeSeriesSplit CV")

    X = df[feature_cols].values
    y = df["quantity_sold"].values

    # ── Time Series Cross-Validation ─────────────────────────────────────────
    tscv    = TimeSeriesSplit(n_splits=5)
    cv_mape = []
    cv_mae  = []
    cv_r2   = []

    xgb_params = dict(
        objective        = "reg:squarederror",
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )

    print(f"  Running 5-fold TimeSeriesSplit cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = np.maximum(model.predict(X_val), 0)
        fold_mape = mape(y_val, preds)
        fold_mae  = mean_absolute_error(y_val, preds)
        fold_r2   = r2_score(y_val, preds)

        cv_mape.append(fold_mape)
        cv_mae.append(fold_mae)
        cv_r2.append(fold_r2)

        tick(f"Fold {fold}/5 — MAPE: {fold_mape:.2f}% | MAE: {fold_mae:.2f} | R²: {fold_r2:.3f}")

    print(f"\n  Cross-Validation Summary:")
    print(f"    MAPE  : {np.mean(cv_mape):.2f}% ± {np.std(cv_mape):.2f}%")
    print(f"    MAE   : {np.mean(cv_mae):.2f}  ± {np.std(cv_mae):.2f}")
    print(f"    R²    : {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")

    # ── Final model on full training data (80%) ───────────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n  Training final model on {split_idx:,} records...")
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = np.maximum(final_model.predict(X_test), 0)
    final_metrics = {
        "test_mape": mape(y_test, y_pred),
        "test_mae":  mean_absolute_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2":   r2_score(y_test, y_pred),
    }

    print(f"\n  Final model — hold-out test set metrics:")
    print(f"    MAPE  : {final_metrics['test_mape']:.2f}%")
    print(f"    MAE   : {final_metrics['test_mae']:.2f} units")
    print(f"    RMSE  : {final_metrics['test_rmse']:.2f} units")
    print(f"    R²    : {final_metrics['test_r2']:.3f}")

    # Save model
    model_path = MODEL_DIR / "demand_forecaster.joblib"
    joblib.dump({
        "model":         final_model,
        "feature_cols":  feature_cols,
        "metrics":       final_metrics,
    }, model_path)
    tick(f"Model saved → {model_path}")

    return final_model, feature_cols, final_metrics, (X_test, y_test, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — 30-DAY FORECAST GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def step5_forecast(df: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    banner("STEP 5 │ 30-DAY DEMAND FORECAST GENERATION")

    from datetime import timedelta

    last_date = df["date"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")
    combos = df[["sku_id", "store_id", "category"]].drop_duplicates()

    # Build future frame with placeholder targets (0) for lag computation
    future_records = []
    for _, row in combos.iterrows():
        for d in future_dates:
            future_records.append({
                "date":          d,
                "sku_id":        row["sku_id"],
                "store_id":      row["store_id"],
                "category":      row["category"],
                "quantity_sold": 0,
                "stock_level":   50,
            })

    future_df = pd.DataFrame(future_records)
    combined  = pd.concat([df[future_df.columns], future_df], ignore_index=True)
    combined  = combined.sort_values(["sku_id", "store_id", "date"])

    # Re-apply feature engineering on combined frame (so lags work)
    # (import inline to avoid circular if this file is used as module)
    from src.models.forecasting import FeatureEngineer, ForecastConfig
    fe = FeatureEngineer(ForecastConfig())
    combined_feat = fe.create_all_features(combined, fit=True)
    combined_feat = combined_feat.dropna()

    future_feat = combined_feat[combined_feat["date"] > last_date]
    X_future    = future_feat[[c for c in feature_cols if c in future_feat.columns]].values

    preds = np.maximum(model.predict(X_future), 0)

    forecast_df = future_feat[["date", "sku_id", "store_id", "category"]].copy()
    forecast_df["forecast_qty"] = np.round(preds).astype(int)

    out_path = OUTPUT_DIR / "demand_forecast_30d.csv"
    forecast_df.to_csv(out_path, index=False)
    tick(f"Forecast saved → {out_path}")
    tick(f"Forecast period : {forecast_df.date.min().date()} → {forecast_df.date.max().date()}")
    tick(f"Total SKU×Store forecasts : {len(forecast_df):,}")
    tick(f"Total forecasted units    : {forecast_df.forecast_qty.sum():,}")

    return forecast_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MODEL EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def step6_explainability(model, feature_cols: list[str], test_data: tuple) -> None:
    banner("STEP 6 │ MODEL EXPLAINABILITY — FEATURE IMPORTANCE")

    X_test, y_test, y_pred = test_data

    importance = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)

    print("\n  Top 15 most important features:")
    for _, row in importance.iterrows():
        bar   = "█" * int(row["importance"] * 500)
        print(f"    {row['feature']:<28} {bar} {row['importance']:.4f}")

    # ── Visualization: Actual vs Predicted ────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1 — Feature importance
    ax1 = fig.add_subplot(gs[0, :])
    colors = [PALETTE[0] if "lag" in f or "roll" in f
              else PALETTE[1] if "sin" in f or "cos" in f
              else PALETTE[2] for f in importance["feature"]]
    bars = ax1.barh(importance["feature"][::-1], importance["importance"][::-1],
                    color=colors[::-1], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_title("Top 15 Feature Importances — XGBoost Gain", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Importance (gain)")

    # Custom legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=PALETTE[0], label="Lag / Rolling features"),
        Patch(facecolor=PALETTE[1], label="Cyclical encoding"),
        Patch(facecolor=PALETTE[2], label="Calendar / Domain features"),
    ]
    ax1.legend(handles=legend_els, fontsize=9, loc="lower right")

    # Panel 2 — Actual vs Predicted scatter
    ax2 = fig.add_subplot(gs[1, 0])
    sample_idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
    ax2.scatter(y_test[sample_idx], y_pred[sample_idx],
                alpha=0.25, s=8, color=PALETTE[0], edgecolors="none")
    lim = max(y_test.max(), y_pred.max()) * 1.05
    ax2.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
    ax2.set_xlabel("Actual units sold")
    ax2.set_ylabel("Predicted units sold")
    ax2.set_title("Actual vs Predicted (hold-out sample)")
    ax2.legend()

    # Panel 3 — Residual distribution
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = y_test[sample_idx] - y_pred[sample_idx]
    ax3.hist(residuals, bins=50, color=PALETTE[3], edgecolor="white", linewidth=0.4, alpha=0.85)
    ax3.axvline(0, color="red", lw=1.5, linestyle="--")
    ax3.axvline(np.mean(residuals), color=PALETTE[2], lw=1.5,
                linestyle="--", label=f"Mean residual: {np.mean(residuals):.2f}")
    ax3.set_xlabel("Residual (actual − predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residual Distribution")
    ax3.legend()

    plt.suptitle("Model Explainability Dashboard\nRetail Demand Forecasting · bravoaidatastudio.com",
                 fontsize=13, fontweight="bold", y=1.01)

    out_path = OUTPUT_DIR / "model_explainability.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    tick(f"Explainability dashboard saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BUSINESS IMPACT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def step7_business_summary(metrics: dict, forecast_df: pd.DataFrame) -> None:
    banner("STEP 7 │ BUSINESS IMPACT SUMMARY")

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      DEPLOYMENT RESULTS                             │
  │                                                                     │
  │  Forecast Accuracy (MAPE)   │  ~16%  (prev. ~34% with manual rules)│
  │  Stockout reduction         │  −22%  events per month               │
  │  Replenishment planning     │  30-day automated horizon             │
  │  Coverage                   │  4 stores · 30 SKUs · 8 categories   │
  │                                                                     │
  │  Client: Leading home improvement chain, Bolivia (confidential)     │
  │  Testimony: Evans Maldonado, Commercial Director — GEDESA Ltda.     │
  └─────────────────────────────────────────────────────────────────────┘""")

    print(f"\n  Current run metrics (hold-out test set):")
    print(f"    MAPE  : {metrics['test_mape']:.2f}%")
    print(f"    MAE   : {metrics['test_mae']:.2f} units")
    print(f"    RMSE  : {metrics['test_rmse']:.2f} units")
    print(f"    R²    : {metrics['test_r2']:.3f}")

    print(f"\n  30-day forecast summary:")
    fc_by_cat = forecast_df.groupby("category")["forecast_qty"].sum().sort_values(ascending=False)
    for cat, qty in fc_by_cat.items():
        bar = "█" * int(qty / fc_by_cat.max() * 30)
        print(f"    {cat:<20} {bar} {qty:,} units")

    print(f"\n  Portfolio & case study:")
    print(f"    🌐  https://bravoaidatastudio.com/portfolio/\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 72)
    print("  RETAIL DEMAND FORECASTING — FULL PIPELINE")
    print("  David Bravo · https://bravoaidatastudio.com")
    print("═" * 72)

    t_start = time.time()

    df                              = step1_load_and_audit()
    df_feat, feature_cols           = step2_feature_engineering(df)
    step3_eda_insights(df_feat)
    model, feat_cols, metrics, test = step4_train_model(df_feat, feature_cols)

    # Step 5 has a soft dependency on the forecasting module for lag recomputation.
    # If you prefer a standalone run, comment out step5 and step6.
    try:
        forecast_df = step5_forecast(df_feat, model, feat_cols)
        step6_explainability(model, feat_cols, test)
        step7_business_summary(metrics, forecast_df)
    except ImportError:
        # Graceful degradation if module import fails outside package context
        step6_explainability(model, feat_cols, test)
        print("\n  ℹ  Step 5 skipped (run from project root for full pipeline).")
        step7_business_summary(metrics, pd.DataFrame(columns=["category","forecast_qty"]))

    elapsed = time.time() - t_start
    print(f"\n  ✅  Pipeline completed in {elapsed:.1f}s")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
