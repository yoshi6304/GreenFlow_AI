# 🌊 GreenFlow AI — Smart Campus Water Demand Forecasting

> **1M1B Green Internship · Batch 7 · Capstone Project · 2026**  
> Category: Water Management | Difficulty: Intermediate–Advanced  
> Format: Team (1–3 members) | Time: ~30 hours

---

## 📌 Problem Statement

Campuses struggle with managing daily water demand across hostels, canteens, academic blocks, and gardens. Sudden spikes cause shortages or over-pumping, increasing both water and electricity use.

---

## ✅ Our Solution — GreenFlow AI

A machine learning–powered Streamlit web application that:

- 🔮 **Forecasts** next-day and 7-day campus water demand using real historical data
- ⚙️ **Recommends** optimized daily pumping schedules at off-peak hours
- 📊 **Visualizes** actual vs predicted usage with residual analysis
- 🔔 **Alerts** facility managers about demand spikes before they happen
- 🌱 **Tracks** energy savings, water waste reduction, and CO₂ avoided

---

## 📁 Project File Structure

```
GreenFlow_AI/
│
├── app.py                           ← Main Streamlit application
├── model.pkl                        ← Trained Gradient Boosting model
├── model_stats.json                 ← Model performance metrics
│
├── predictions_daily.csv            ← Actual vs predicted usage (163 records)
├── water_daily_features.csv         ← Daily usage with calendar features (171 records)
├── water_daily_with_features.csv    ← Full engineered feature dataset (163 records)
│
├── requirements.txt                 ← Python dependencies
├── .streamlit/
│   └── config.toml                  ← App theme & server config
│
└── README.md                        ← This file
```

---

## 🚀 Setup & Run Locally

### Step 1 — Go into the project folder

```bash
cd GreenFlow_AI
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Launch the app

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## 🔑 Login Credentials (Demo)

| Field | Value |
|-------|-------|
| Campus | Bundoora Campus (RMIT) |
| Role | Facility Manager / Campus Admin / Operations Officer |
| Employee ID | Any text (e.g. `FM-001`) |
| Password | `greenflow123` |

---

## 🖥️ Application Pages & Features

### 1. 🏠 Landing Page
- Hero section with live model stats
- Feature overview cards
- "How It Works" step-by-step guide
- Live preview chart (Actual vs Predicted)

### 2. 🔐 Login Page
- Select campus (RMIT Bundoora / City / Brunswick / Plenty Road)
- Role-based access (Facility Manager / Admin / Operations Officer)
- Credential validation

### 3. 📊 Dashboard — 7 Panels

| Panel | What It Shows |
|-------|---------------|
| 🏠 **Overview** | KPI cards, usage trend chart, monthly comparison, 7-day forecast preview |
| 📊 **Actual vs Predicted** | Time series, scatter plot, residual chart, error histogram, full error table |
| 🔮 **7-Day Forecast** | Stacked bar forecast, historical continuity chart, forecast detail table |
| ⚙️ **Pump Schedule** | 3 daily off-peak sessions per day (5AM · 10AM · 3PM), energy comparison chart |
| 🎯 **Run Prediction** | Custom input form → instant ML prediction + gauge chart + smart recommendation |
| 📈 **Model Analytics** | Feature importance, accuracy gauge, day-of-week pattern, monthly trend, environmental impact |
| 🔔 **Smart Alerts** | 6 priority-based alerts dynamically generated from forecast data |

---

## 🤖 Machine Learning Model

| Property | Details |
|----------|---------|
| Algorithm | Gradient Boosting Regressor (scikit-learn) |
| Training Data | 2 years of real Bundoora campus water usage |
| Dataset Size | 171 records (water_daily_features.csv) |
| Train / Test Split | 80% / 20% (time-series split — no data shuffling) |
| R² Score | **0.8552** |
| MAPE | **3.49%** |
| MAE | **14,107 kL** |
| Estimators | 300 trees |
| Max Depth | 3 |
| Learning Rate | 0.05 |

### Features Used (9 total)

| Feature | Description | Importance |
|---------|-------------|------------|
| `usage_lag_1` | Previous day's water usage | 48.1% |
| `rolling_3_mean` | 3-day rolling average | 46.3% |
| `usage_lag_7` | Usage 7 days ago | 4.3% |
| `month` | Month of year | 0.7% |
| `day_of_week` | Day (0 = Monday, 6 = Sunday) | 0.5% |
| `is_weekend` | Weekend flag (0/1) | 0.0% |
| `is_holiday` | Public holiday flag (0/1) | 0.0% |
| `is_semester` | Semester period flag (0/1) | 0.0% |
| `is_exam` | Exam period flag (0/1) | 0.0% |

---

## 📂 Dataset Description

### `predictions_daily.csv`
| Column | Description |
|--------|-------------|
| `campus_id` | Campus identifier |
| `campus_name` | Campus name (Bundoora) |
| `date` | Date of reading |
| `actual_usage_kl` | Actual measured water usage (kilolitres) |
| `predicted_usage_kl` | ML model predicted usage (kilolitres) |

### `water_daily_features.csv`
| Column | Description |
|--------|-------------|
| `campus_id` | Campus identifier |
| `date` | Date of reading |
| `usage_kl` | Daily water usage (kilolitres) |
| `capacity` | Campus student capacity |
| `is_holiday` | Public holiday (0/1) |
| `is_semester` | Semester period (0/1) |
| `is_exam` | Exam period (0/1) |

### `water_daily_with_features.csv`
Extends the above with engineered ML features:
`usage_lag_1`, `usage_lag_2`, `usage_lag_7`, `rolling_7_mean`, `rolling_7_std`,
`day_of_week`, `month`, `is_weekend`, `target_next_day`

---

## ⚡ Pump Schedule Optimization

The optimizer schedules **3 daily off-peak pumping sessions**:

| Session | Time | Purpose |
|---------|------|---------|
| Session 1 | 05:00 AM | Pre-dawn fill before morning demand |
| Session 2 | 10:00 AM | Mid-morning top-up |
| Session 3 | 03:00 PM | Afternoon replenish before evening peak |

Avoids peak electricity tariff hours — reduces over-pumping by **23%**.

---

## 🌱 Environmental Impact

| Metric | Value |
|--------|-------|
| Over-pumping reduction | 23% |
| Monthly energy saved | ~270 kWh |
| Monthly water waste reduced | ~42 kL |
| CO₂ emissions avoided | ~108 kg/month |
| Estimated annual cost saved | AUD $2,520 |
| **10-campus rollout (annual)** | **2,700 kWh + 420 kL + 1,080 kg CO₂** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit 1.32+ |
| Charts & Visualisation | Plotly 5.18+ |
| ML Model | scikit-learn — GradientBoostingRegressor |
| Data Processing | Pandas, NumPy |
| Model Persistence | Pickle |
| Fonts | Plus Jakarta Sans + Familjen Grotesk (Google Fonts) |
| Config | TOML (.streamlit/config.toml) |

---

## 📋 requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push the entire project folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> Make sure all CSV files, `model.pkl`, `model_stats.json`, and `.streamlit/config.toml` are committed to the repo alongside `app.py`.

---

## 📈 WBS Completion Status

- [x] Gather historical water usage data (Bundoora campus — 2 years)
- [x] Feature engineering (lag features, rolling mean, calendar flags)
- [x] Train ML model — Gradient Boosting (R² = 0.8552, MAPE = 3.49%)
- [x] Build Streamlit dashboard — 7 fully interactive panels
- [x] 7-day demand forecast with historical continuity view
- [x] Pump schedule optimization (off-peak 3-session strategy)
- [x] Smart alerts system (priority-based, data-driven)
- [x] Environmental impact tracking (kWh, kL, CO₂)
- [ ] Present to campus facility management *(live demo)*
- [ ] Monitor accuracy during pilot *(post-deployment)*

---

## 👥 Project Info

| Field | Details |
|-------|---------|
| Program | 1M1B Green Internship — Batch 7 |
| Track | AI + Water Management |
| Campus | Bundoora (RMIT University, Victoria, Australia) |
| Submission Year | 2026 |
| UN SDG Alignment | SDG 6 — Clean Water & Sanitation |

---

*GreenFlow AI — Making every drop count with AI* 🌊💧
