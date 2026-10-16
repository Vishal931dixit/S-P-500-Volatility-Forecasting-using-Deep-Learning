# 📈 S&P 500 Volatility Forecasting — Deep Learning & Econometric Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![ARCH](https://img.shields.io/badge/ARCH-Econometrics-green.svg)](https://arch.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-yellow.svg)](https://scikit-learn.org/)
[![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-red.svg)](https://pypi.org/project/yfinance/)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Reproducible-brightgreen.svg)]()

---

## 📘 Executive Summary

Market volatility is the heartbeat of financial risk management, derivative pricing, and quantitative portfolio allocation. The CBOE Volatility Index (**VIX**)—often termed the market's *"fear gauge"*—reflects implied 30-day volatility derived from S&P 500 options, but actively forecasting future **realized volatility** remains one of quantitative finance's most complex challenges due to volatility clustering, heavy tails, and regime shifts.

This project delivers an **end-to-end quantitative modeling pipeline** comparing **classical econometrics** against **modern recurrent deep learning architectures** on nearly **15 years of daily S&P 500 (`^GSPC`) and VIX (`^VIX`) data** (March 2005 – December 2019, 3,726 trading sessions). 

### 🏆 Key Achievements & Quantitative Highlights
* **Leakage-Free Feature Engineering:** Engineered **70 technical and statistical features**, identified and purged a subtle **0.9871 correlation data-leakage variable**, and enforced a strict **80/10/10 chronological split**.
* **Deep Learning Superiority:** Optimized Gated Recurrent Units (**GRU**) and Long Short-Term Memory (**LSTM**) networks achieved **$R^2 \approx 0.79 - 0.82$** and **Test RMSE $\approx 8.25 - 9.07$**, substantially outperforming classical GARCH ($R^2 = 0.69$) and EWMA ($R^2 = 0.52$).
* **Alpha Generation via Directional Accuracy:** The optimized recurrent network achieved **71.1% directional accuracy** on out-of-sample data, providing a viable alpha signal for delta-neutral volatility trading (options straddles/strangles).
* **The "Horizontal Flatline" GRU Case Study:** Thoroughly diagnosed and resolved the classic deep learning failure mode where recurrent models collapse into predicting the global sample mean ($\sigma_{\text{pred}} \approx 1.29$ vs $\sigma_{\text{actual}} \approx 19.62$).

---

## 🏗️ End-to-End Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1. DATA INGESTION                                      │
│  Daily S&P 500 (^GSPC) & CBOE Volatility Index (^VIX) via Yahoo Finance (2005 - 2019)  │
│                     3,726 Raw Daily Trading Sessions                                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                            2. FEATURE ENGINEERING (70 Features)                        │
│  • Rolling Volatilities (5d, 10d, 20d, 50d)     • Parkinson & Garman-Klass Volatility  │
│  • Log Returns, Squared Returns, Vol of Vol     • VIX Spread & VIX_Vol_Ratio           │
│  • Normalized RSI, Price-to-MA Ratios           • Multi-Period Temporal Lags (t-1..10) │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                       3. LEAKAGE AUDIT & MULTICOLLINEARITY FIX                         │
│  • Purged raw OHLC collinearity ($r \approx 1.0$)                                      │
│  • Discovered and dropped `Close_rolling_std_20` ($r = 0.9871$ target leakage)        │
│  • StandardScaler fit strictly on Training Set (preventing forward lookahead bias)     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                       4. CHRONOLOGICAL 80 / 10 / 10 SPLITTING                          │
│  • Train: 2,940 days (Mar 2005 – Aug 2016) [2008 GFC & Flash Crash, Mean Vol: 24.51]   │
│  • Val:     368 days (Aug 2016 – Feb 2018) [2017 Bull Run & Feb 2018 Volmageddon]     │
│  • Test:    368 days (Feb 2018 – Dec 2019) [Q4 2018 Market Sell-off, Mean Vol: 41.37] │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                             5. MODEL TOURNAMENT ROSTER                                 │
│  ┌───────────────────────┬───────────────────────┬──────────────────────────────────┐  │
│  │ Classical Econometric │  Vanilla Recurrent    │  Modern Gated Deep Learning      │  │
│  ├───────────────────────┼───────────────────────┼──────────────────────────────────┤  │
│  │ • EWMA ($\lambda=0.94$)│ • Tuned RNN (16 units,│ • GRU (Fast, Medium, Deep tiers) │  │
│  │ • GARCH(1,1)          │   tanh, 20% dropout,  │ • LSTM (Fast, Medium, Deep tiers)│  │
│  │ • EGARCH(1,1) (leverage│   15d sequence window)│ • Diagnosed & fixed Flatline Bug │  │
│  └───────────────────────┴───────────────────────┴──────────────────────────────────┘  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│                       6. EVALUATION & TRADING APPLICATION                              │
│  • Metrics: Test RMSE, Out-of-Sample $R^2$, Directional Accuracy (71.1%)               │
│  • Quant Applications: Volatility Arbitrage, Straddle Timing, Dynamic VaR Sizing       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Target Formulation

### 1. Primary Forecasting Target — 20-Day Annualized Realized Volatility
While the VIX index measures market *expectations* of 30-day volatility, the prediction target for all regression and deep learning models in this study is the **20-day annualized realized volatility of the S&P 500**:

$$\text{Log\_Return}_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

$$\text{Volatility}_t = \text{std}\left(\text{Log\_Return}_{t-19 \dots t}\right) \times \sqrt{252} \times 100$$

Where $252$ is the standard number of trading days in a U.S. financial year, expressing volatility in annualized percentage terms.

| Statistic | Full Dataset | Training Set (80%) | Out-of-Sample Test Set (10%) |
|---|---|---|---|
| **Observations** | 3,726 days | 2,940 days | 368 days |
| **Date Range** | Mar 2005 – Dec 2019 | Mar 2005 – Aug 2016 | Feb 2018 – Dec 2019 |
| **Mean Volatility** | 24.51% | 22.40% | **41.37%** |
| **Std Deviation** | 15.21% | 14.15% | **19.62%** |
| **Min Volatility** | 5.02% | 5.02% | 15.12% |
| **Max Volatility** | 122.96% | 122.96% | 122.96% |

> [!IMPORTANT]
> **Market Regime Shift in Test Horizon:** The test set represents a historically turbulent market regime (average volatility of **41.37%** vs **22.40%** in training) driven by the Q4 2018 Fed rate hike correction and the 2019 US-China trade dispute. This structural shift explains why Test RMSE values are naturally higher than training RMSE across all models.

---

### 2. Feature Engineering Taxonomy (70 Input Features)
Raw OHLC price feeds carry severe multicollinearity ($r \approx 1.0$). To build robust predictors, 70 features were engineered across four mathematical categories:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            70 ENGINEERED INPUT FEATURES                          │
├─────────────────────┬────────────────────┬───────────────────┬───────────────────┤
│ Return Dynamics     │ Volatility Scales  │ VIX & Implied Vol │ Technical & Lags  │
├─────────────────────┼────────────────────┼───────────────────┼───────────────────┤
│ • Daily Log Return  │ • Rolling Std 5d   │ • Log_VIX         │ • RSI Normalized  │
│ • Squared Returns   │ • Rolling Std 10d  │ • VIX_Vol_Ratio   │ • Price_MA_Ratio  │
│ • Absolute Returns  │ • Rolling Std 50d  │   (Implied vs     │ • MA_Spread       │
│ • Vol of Volatility │ • Parkinson Vol    │    Realized)      │ • Bollinger Bands │
│ • Return/Vol Ratio  │ • Garman-Klass Vol │ • VIX 5d/20d Lags │ • Lags: t-1..t-10 │
└─────────────────────┴────────────────────┴───────────────────┴───────────────────┘
```

---

## 🛡️ The Data Leakage Audit & Multicollinearity Fix

In financial machine learning, data leakage is the primary cause of models that look brilliant in notebooks but collapse in live trading. This project implemented two critical defensive audits:

### 1. The 0.9871 Correlation Discovery (Cell 12 Audit)
During feature correlation auditing against the target variable, a critical threat was identified:
* `Close_^GSPC_rolling_std_20` displayed a **0.9871 Pearson correlation** with `Volatility`.
* **Root Cause:** The 20-day rolling standard deviation of the close price tracks price dispersion almost identically to the rolling standard deviation of log returns. Leaving this feature in would create a trivial identity mapping, leaking target information directly into the inputs.
* **Resolution:** `Close_^GSPC_rolling_std_20` was permanently pruned, reducing the finalized feature roster from 70 to 69 before model training.

### 2. Temporal Preprocessing Integrity
* Normalization was performed via `StandardScaler`.
* The scaler was fitted **strictly on `X_train`** (the first 2,940 samples). 
* The training mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ were then applied to scale `X_val` and `X_test`, ensuring zero future information leaked into historical transformations.

---

## 🔬 Model Architectures & Formulations

### 1. Classical Econometric Baselines
* **EWMA (Exponentially Weighted Moving Average):**
  $$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2$$
  * Tested decay factors from $\lambda = 0.85$ to $0.99$. The RiskMetrics market-standard $\lambda = 0.94$ proved optimal (Test RMSE: 13.97, $R^2: 0.52$).
* **GARCH(1,1) & EGARCH(1,1):**
  $$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2 \quad \text{subject to } \alpha + \beta < 1$$
  * Models volatility clustering and persistence. EGARCH incorporates the leverage effect—where negative equity shocks trigger greater volatility expansion than positive returns of equal magnitude.

### 2. Tuned Vanilla RNN (HPO Optimized)
* Sequential grid search over unit sizes $\{8, 16, 32\}$, layers $\{1, 2\}$, sequence lengths $\{5, 10, 15, 20\}$, and dropout rates $\{0.1, 0.2, 0.3\}$.
* **Optimal Topology:** Single recurrent layer with 16 units, $\tanh$ activation, 20% dropout, lookback sequence $L = 15$ days, Adam optimizer ($\eta = 0.005$).
* **Outcomes:** Test RMSE: 10.96 | Test $R^2$: 0.687 | **Directional Accuracy: 71.1%**.

### 3. Gated Deep Learning: GRU & LSTM
* **Gated Recurrent Unit (GRU):** Utilizes update ($z_t$) and reset ($r_t$) gates to regulate long-term memory flow with fewer parameters than LSTM:
  $$z_t = \sigma(W_z x_t + U_z h_{t-1}), \quad r_t = \sigma(W_r x_t + U_r h_{t-1})$$
  $$\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1})), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
* **Long Short-Term Memory (LSTM):** Employs input ($i_t$), forget ($f_t$), and output ($o_t$) gates with a dedicated Cell State ($C_t$) Constant Error Carousel (CEC) to eliminate gradient vanishing over the 15-day temporal window.

---

## 🐛 Case Study: Diagnosing the "Horizontal Flatline" GRU Bug

One of the most valuable software engineering takeaways in this project occurred during initial GRU experimentation (documented in Notebook Cells 34–36):

```
Initial GRU Predictions:  Min: 28.83  | Max: 37.12  | Mean: 33.46  | Std Dev: 1.29 ❌ (FLATLINE)
Actual Test Volatility:   Min: 15.12  | Max: 122.96 | Mean: 41.37  | Std Dev: 19.62 ✅
```

### 🔍 Diagnostic Root Cause Analysis
1. **The Phenomenon:** The initial deep 3-layer GRU produced a nearly flat horizontal line on the test set. Its predicted standard deviation was only **1.29**, while actual market volatility fluctuated between 15 and 123 ($\sigma = 19.62$).
2. **Why It Happened:** 
   * When stacking 3 dense GRU layers on noisy financial sequences, gradients vanished across the recurrent transitions.
   * Driven by mean squared error loss, the model found a comfortable local minimum by predicting the **unconditional sample mean ($\approx 33.5$)**, minimizing risk across outliers rather than learning temporal dynamics.
3. **The Engineering Fix:**
   * Reduced model depth to 1–2 layers, matching the proven parameter density of the successful RNN.
   * Standardized lookback sequences strictly to $L=15$ trading days.
   * Added 20% dropout regularization and standardized gradient updates with Adam ($\text{lr} = 0.005$).
   * **Result:** Corrected `GRU_Medium_v2` achieved **Test $R^2 = 0.786$ and RMSE = 9.07**, dynamically tracking the full peak-to-trough range of the test set.

---

## 🏆 Comprehensive Model Leaderboard

All models were evaluated on the identical 368-day out-of-sample test horizon (February 2018 – December 2019):

| Rank | Model Architecture | Train $R^2$ | Val $R^2$ | Test $R^2$ | Test RMSE | Directional Accuracy | Training Time | Model Category |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 🥇 | **LSTM_Fast** (16 units, 1L) | **0.9557** | **0.8650** | **0.8225** | **8.2551** | ~69.4% | 15.1s | Deep Learning (Gated) |
| 🥈 | **GRU_Fast_v2** (16 units, 1L) | 0.9722 | 0.8410 | **0.8055** | **8.6407** | ~70.2% | 14.4s | Deep Learning (Gated) |
| 🥉 | **GRU_Medium_v2** (32 units, 2L) | 0.9373 | 0.8120 | **0.7858** | **9.0672** | ~68.8% | 29.2s | Deep Learning (Gated) |
| 4 | **LSTM_Medium** (32 units, 2L) | 0.9327 | 0.8050 | 0.7724 | 9.3484 | ~67.5% | 30.0s | Deep Learning (Gated) |
| 5 | **LSTM_Simple** (16 units, 1L) | 0.9647 | 0.7910 | 0.6132 | 12.1853 | ~65.1% | 15.9s | Deep Learning (Gated) |
| 6 | **RNN_HPO_Optimal** (16 units, 1L) | 0.9479 | 0.8562 | 0.6873 | 10.9564 | **71.1%** | 0.02s | Vanilla Recurrent |
| 7 | **GARCH(1,1)** | — | — | 0.6912 | 10.9600 | 54.2% | <1.0s | Econometric Statistical |
| 8 | **EWMA** ($\lambda = 0.94$) | 0.9687 | 0.7100 | 0.5210 | 13.9700 | 51.6% | Instant | Classical Smoothing |

---

## 💼 Quantitative Finance & Trading Applications

### 1. Volatility Arbitrage (VRP Trading)
The Volatility Risk Premium (**VRP**) is defined as the spread between option-implied volatility ($VIX$) and forecasted realized volatility ($\hat{\sigma}$):
$$\text{VRP}_t = VIX_t - \hat{\sigma}_{t, t+20}$$
When our GRU/LSTM model forecasts realized volatility significantly below current $VIX$, implied volatility is overpriced, creating a statistical edge for **short variance / delta-neutral options straddle selling**.

### 2. High Directional Accuracy (71.1%) as an Execution Alpha
In options market-making, predicting the exact numerical volatility level is secondary to knowing **direction**:
* If the model predicts volatility will expand tomorrow ($\hat{\sigma}_{t+1} > \sigma_t$), market makers widen bid-ask spreads and buy long gamma (long straddles).
* Achieving **71.1% directional accuracy** with the tuned RNN substantially exceeds the 50% random-walk baseline, providing a statistically significant directional trading signal.

### 3. Dynamic Value-at-Risk (VaR) Sizing
Traditional risk engines rely on static normal distributions or slow-moving EWMA models. Feeding daily GRU-forecasted volatility into 99% 1-day Value-at-Risk calculations:
$$\text{VaR}_{99\%} = 2.326 \times \hat{\sigma}_{\text{GRU}, t+1} \times \text{Portfolio Value}$$
This allows risk managers to dynamically increase hedge ratios *before* market sell-offs strike, reducing maximum drawdown.

---

## 🛠️ Project Structure & Reproduction

### Repository Structure
```
├── README.md                      # Comprehensive project documentation
├── VolatilityModelling.ipynb     # Complete 39-cell analysis and training notebook
├── main.tex                      # Professional resume documenting project outcomes
└── model_plots/                  # Visual artifacts generated from training runs
    ├── RNN_HPO_Optimal_time_series.png
    ├── GRU_Medium_v2_time_series.png
    ├── LSTM_Fast_time_series.png
    ├── GARCH_volatility_plot.png
    └── correlation_heatmaps.png
```

### Installation & Execution
```bash
# 1. Clone the repository
git clone https://github.com/ashwini0008/S-P-500-VOLATILITY-FORECASTING.git
cd S-P-500-VOLATILITY-FORECASTING

# 2. Install required dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow>=2.20.0 arch yfinance

# 3. Launch the Jupyter Notebook
jupyter notebook VolatilityModelling.ipynb
```

---

## ⚖️ License & Attribution

Developed by **Vishal Dixit** (M.Tech Management Sciences, IIT Kanpur).  
Market data sourced from Yahoo Finance (`yfinance`). Released under the MIT License.
