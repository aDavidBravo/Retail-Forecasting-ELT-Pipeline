# Retail Demand Forecasting & ELT Pipeline

> **End-to-end forecasting system for a leading home improvement retailer in Bolivia** — reducing stockouts by 22% and improving demand accuracy by 18% through a modern ELT architecture and XGBoost-based ML pipeline.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost" />
  <img src="https://img.shields.io/badge/dbt-1.7-FF694B?logo=dbt&logoColor=white" />
  <img src="https://img.shields.io/badge/Snowflake-Data%20Warehouse-29B5E8?logo=snowflake&logoColor=white" />
  <img src="https://img.shields.io/badge/Airbyte-ELT-615EFF?logo=airbyte&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📌 Project Overview

This repository contains the data engineering and machine learning pipeline built for a confidential home improvement retail chain operating across three major Bolivian cities (La Paz, Santa Cruz, Cochabamba). The system ingests daily transactional data, transforms it through a layered dbt model, and generates 30-day demand forecasts at the SKU × Store level.

> 🌐 **Full project details and case study available at:** [bravoaidatastudio.com](https://bravoaidatastudio.com/portfolio/)

**The business challenge:** Manual inventory replenishment decisions were causing ~22% of lost sales due to stockouts, with no visibility into seasonal demand patterns tied to Bolivia's rainy/dry season cycle or national holidays.

---

## 📊 Business Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Forecast Accuracy (MAPE) | ~34% error | ~16% error | **+18 pp improvement** |
| Stockout Events / Month | ~47 avg | ~37 avg | **−22% reduction** |
| Replenishment Lead Time | Reactive (manual) | Proactive (30-day horizon) | **Automated** |
| Cities Covered | 3 | 3 + El Alto | **+1 distribution center** |

---

## 🏗️ Architecture

The system follows a modern **ELT** pattern, decoupling ingestion, transformation, and prediction into independently scalable layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   POS System  │  Inventory DB  │  Supplier Feeds  │  Calendar  │
└───────┬────────────────┬───────────────┬──────────────┬─────────┘
        │                │               │              │
        ▼                ▼               ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION  (Airbyte)                          │
│         Incremental CDC sync → Snowflake RAW layer              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSFORMATION  (dbt)                           │
│   Staging → Intermediate → Marts (fct_sales, dim_products...)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ML FORECASTING  (Python / XGBoost)              │
│   Feature Engineering → Time-Series CV → 30-day SKU forecast    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CONSUMPTION  (Power BI / Snowflake)             │
│         Executive Dashboards  │  Replenishment Alerts            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline

### Feature Engineering

The model was built with careful domain knowledge of the Bolivian retail calendar:

| Feature Group | Examples |
|---|---|
| **Temporal** | Day of week, week of year, month, quarter |
| **Cyclical encoding** | sin/cos of day, week, month (preserves periodicity) |
| **Bolivian seasonality** | Rainy season flag (Nov–Mar), dry season flag (Apr–Oct) |
| **Local holidays** | Día del Mar, Año Nuevo Aymara, Independencia, Día del Trabajo |
| **Lag features** | Sales at t−1, t−7, t−14, t−21, t−28 |
| **Rolling statistics** | Mean, std, max, min over 7/14/30-day windows |
| **Store attributes** | City, store size multiplier |

### Model Selection & Validation

- **Algorithm:** XGBoost Regressor (tuned with `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`)
- **Validation strategy:** `TimeSeriesSplit` (5 folds) — no data leakage
- **Primary metric:** MAPE (Mean Absolute Percentage Error)
- **Regularization:** L1 (`reg_alpha=0.1`) + L2 (`reg_lambda=1.0`) to prevent SKU-level overfitting

---

## 📂 Repository Structure

```
Retail-Forecasting-ELT-Pipeline/
├── dbt_project/                  # SQL transformation models
│   ├── models/
│   │   ├── staging/              # Raw → cleaned layer
│   │   │   └── stg_sales.sql
│   │   └── marts/                # Business-ready data marts
│   │       ├── fct_sales.sql
│   │       ├── dim_products.sql
│   │       └── dim_stores.sql
│   └── tests/                    # dbt data quality tests
├── notebooks/
│   └── 01_EDA_retail_demand.ipynb   # Full exploratory analysis
├── src/
│   ├── generate_synthetic_data.py   # Reproducible data generation
│   ├── ingestion/                   # Airbyte connector configs
│   └── models/
│       └── forecasting.py           # Core ML pipeline (DemandForecaster)
├── data/                            # Generated datasets (gitignored in prod)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone & install**
```bash
git clone https://github.com/aDavidBravo/Retail-Forecasting-ELT-Pipeline.git
cd Retail-Forecasting-ELT-Pipeline
pip install -r requirements.txt
```

**2. Generate the dataset**
```bash
python src/generate_synthetic_data.py
# → outputs data/raw_sales.csv (~130k rows, 3 years of daily SKU×Store sales)
```

**3. Run the EDA notebook**
```bash
jupyter notebook notebooks/01_EDA_retail_demand.ipynb
```

**4. Train the forecasting model and generate predictions**
```bash
python src/models/forecasting.py
# → outputs data/demand_forecast_30d.csv + saves model to models/
```

---

## 📈 Key Findings from EDA

- **Saturday** is the highest-demand day across all categories (+40% vs weekday average)
- **Paint & Waterproofing** sales peak in November–December (onset of rainy season) — up to 2.1× baseline
- **Construction materials** show inverse pattern, peaking in June–August (dry season)
- **Agosto 6 (Independence Day)** generates the single highest daily demand spike of the year (+60%)
- Compound annual growth of ~6.5% across all SKUs reflects Bolivia's growing construction sector

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Warehouse | Snowflake |
| ELT Ingestion | Airbyte |
| Transformation | dbt (Data Build Tool) |
| ML Framework | XGBoost, scikit-learn |
| Data Processing | pandas, NumPy |
| Model Persistence | joblib |
| Notebook Analysis | Jupyter, Matplotlib, Seaborn |

---

## 👤 Author

**David Bravo** — Data Scientist & AI Solutions Architect

> 🌐 [bravoaidatastudio.com](https://bravoaidatastudio.com) | 📁 [Portfolio](https://bravoaidatastudio.com/portfolio/)

*This project was developed under a confidentiality agreement. Business metrics and operational details have been validated by the client. Data in this repository is synthetic and statistically representative of production patterns.*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
