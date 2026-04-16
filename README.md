# 📈 Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assessment

> Analyzing how Bitcoin Fear/Greed sentiment relates to trader behavior and performance on Hyperliquid — uncovering patterns that inform smarter trading strategies.

---

## 🚀 Live Dashboard
👉 https://deval-primetradeai-assessment-round0.streamlit.app/

> Includes sentiment analysis, individual trader explorer, and live strategy signal generator.

---

## 🚀 Quick Summary

| Metric | Value |
|---|---|
| Traders analyzed | 32 unique Hyperliquid accounts |
| Total trades | 211,218 |
| Closed trades analyzed | 104,402 |
| Date range | May 2023 – May 2025 (479 days) |
| Sentiment regimes | 5 (Extreme Fear → Extreme Greed) |
| Coins traded | 246 unique assets |
| Charts produced | 9 |
| Trader archetypes | 4 (KMeans clustering) |
| Predictive model accuracy | 64.7% ± 1.6% (vs 50% baseline) |

---

## 📁 Project Structure

```
primetrade-assessment/
├── data/
│   ├── fear_greed_index.csv          # Bitcoin Fear/Greed index (2018–2025)
│   └── historical_data.csv           # Hyperliquid trader data (2023–2025)
├── charts/
│   ├── chart1_performance_by_sentiment.png
│   ├── chart2_behavior_by_sentiment.png
│   ├── chart3_segmentation.png
│   ├── chart4_heatmap_trader_sentiment.png
│   ├── chart5_trader_rankings.png
│   ├── chart6_bonus_model.png
│   ├── chart7_enhanced_model.png
│   ├── chart8_clustering.png
│   └── chart9_archetype_sentiment.png
├── primetrade_sentiment_analysis.ipynb   # Main analysis notebook
└── README.md
```

---

## ▶️ How to Run

### Option 1 — Google Colab (Recommended, no setup needed)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** → upload `primetrade_sentiment_analysis.ipynb`
3. In the left sidebar, upload both CSV files from the `data/` folder
4. Click **Runtime → Run all**

### Option 2 — Local (Jupyter Notebook)
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook primetrade_sentiment_analysis.ipynb
```

> **Note:** Place both CSV files in the same directory as the notebook before running.

---

## 📊 Analysis Structure

| Section | Description |
|---|---|
| Section 1 | Data Loading — shape, columns, first look |
| Section 2 | Data Quality Check — types, nulls, duplicates |
| Section 3 | Cleaning & Merging — date alignment, inner join on date |
| Section 4 | Core Metrics — PnL, win rate, long/short bias, daily summary |
| Section 5 | Fear vs Greed Performance Analysis |
| Section 6 | Behavioral Analysis — trade size, directional bias, fees |
| Section 7 | Trader Segmentation — frequency, consistency, style |
| Section 8 | Heatmap — every trader × every sentiment regime |
| Section 9 | Trader Rankings per sentiment regime |
| Section 10 | Bonus — Predictive Model (Random Forest) |
| Section 11 | Clustering — 4 behavioral archetypes (KMeans) |
| Section 12 | Executive Summary |

---

## 🔍 Key Findings

### Finding 1 — Contrarian Traders Earn 3.1x More
Only 4 of 32 traders (12.5%) follow a contrarian strategy — going **Long during Fear** and **Short during Greed**. These traders earn an average of **$803,818 total PnL** compared to **$259,879** for the 27 trend-following traders.

The long/short bias data confirms this systematically:
- During Fear: **62–69% of trades are Long** (buying the dip)
- During Greed: **55–58% of trades are Short** (fading the rally)

### Finding 2 — Fear Days Drive Maximum Activity and Strong Returns
Despite being panic days, Fear produces:
- **Highest trade volume:** 29,808 trades (vs 20,853 on Extreme Greed)
- **Largest average trade sizes:** $7,816 (vs $3,112 on Extreme Greed)
- **Second-highest average PnL:** $112.6/trade

Traders treat Fear as **opportunity**, not threat.

### Finding 3 — Sentiment is a Weak Predictor; Trader Momentum is Strong
The predictive model ranks `sentiment_score` **last** in feature importance (0.012). What actually predicts next-day profitability:

| Rank | Feature | Importance |
|---|---|---|
| 1 | profit_streak (consecutive profitable days) | 0.179 |
| 2 | rolling_wr_3d (3-day win rate trend) | 0.125 |
| 3 | total_pnl (today's PnL) | 0.086 |
| 4 | rolling_pnl_3d (3-day momentum) | 0.078 |
| 9 | sentiment_score | 0.012 |

**Market mood alone does not determine outcomes — skill and consistency do.**

---

## 👤 Trader Archetypes (KMeans Clustering, k=4)

| Archetype | Count | Avg Total PnL | Avg Win Rate | Key Characteristic |
|---|---|---|---|---|
| 🎯 Precision Trader | 1 | $199,506 | 100.0% | Only 90 trades, never loses, $3,154/trade on Fear |
| 🐋 Whale Trader | 7 | $897,860 | 85.0% | $16,231 avg trade size, sophisticated capital |
| ⚙️ Consistent Performer | 9 | $216,804 | 97.3% | Near-perfect win rate, thrives on Greed days |
| 📈 Developing Trader | 15 | $121,248 | 76.7% | Struggles on Fear days ($7/trade), most room to improve |

---

## 🎯 Strategy Recommendations (Part C)

### Rule 1 — Sentiment-Adaptive Position Sizing
> **"During Fear (FG index 25–49), traders with a historical Fear win rate
> above 80% should increase position size by up to 40% relative to their
> Neutral-day baseline. Developing Traders (win rate below 80% in Fear)
> should reduce size by 50% — their average PnL drops to just $7/trade
> during Fear vs $130/trade for top performers."**

**Rationale:** Fear days show the largest average trade sizes ($7,816) and
87.3% win rates — data supports aggressive deployment during panic, but
only for traders with a proven Fear-day track record. One size does not
fit all sentiment regimes.

### Rule 2 — Contrarian Trigger System
> **"Activate Long bias when the FG index drops below 30 (deep Fear).
> Activate Short bias when the FG index rises above 75 (Extreme Greed).
> Maintain this directional bias until the FG index returns to Neutral
> (45–55 range)."**

**Rationale:** The 4 contrarian traders using this logic earn 3.1x more
than the 27 trend followers. The thresholds (30 and 75) correspond to the
Fear/Greed index boundaries where directional flip behavior is most
consistently observed in this dataset.

---

## 🤖 Predictive Model (Bonus)

**Goal:** Predict whether a trader will be profitable the next day,
given today's behavior and current market sentiment.

| Metric | Base Model | Enhanced Model |
|---|---|---|
| Algorithm | Random Forest | Random Forest |
| Trees | 100 | 200 |
| Features | 9 | 14 |
| CV Accuracy | 64.7% ± 2.1% | 64.7% ± 1.6% |
| F1 Score | 0.711 ± 0.015 | 0.700 ± 0.012 |
| Random baseline | 50% | 50% |

**Key model finding:** Sentiment ranks last in feature importance.
Trader behavioral momentum (win rate trend, profit streak, PnL momentum)
is far more predictive than market mood.

**Honest limitations:**
- Trained on 32 traders only — needs more data to generalize
- 64.7% accuracy means 35% of predictions are wrong
- Demo-ready, not production-ready without further validation

---

## 📉 Performance by Sentiment Regime

| Sentiment | Avg PnL/Trade | Win Rate | Total Trades | Avg Trade Size |
|---|---|---|---|---|
| Extreme Fear | $71.0 | 76.2% | 10,406 | $5,350 |
| Fear | $112.6 | 87.3% | 29,808 | $7,816 |
| Neutral | $71.2 | 82.4% | 18,159 | $4,783 |
| Greed | $85.4 | 76.9% | 25,176 | $5,737 |
| Extreme Greed | $130.2 | 89.2% | 20,853 | $3,112 |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| pandas | Data manipulation and merging |
| numpy | Numerical operations |
| matplotlib | Chart generation |
| seaborn | Statistical visualizations |
| scikit-learn | Random Forest, KMeans, PCA, cross-validation |
| Google Colab | Development environment |

---

## 👨‍💻 Author

Assessment submission for **Primetrade.ai — Data Science Intern Role**

*Analysis covers Part A (Data Preparation), Part B (Analysis),
Part C (Strategy Recommendations), and both bonus items
(Predictive Model + Trader Clustering).*