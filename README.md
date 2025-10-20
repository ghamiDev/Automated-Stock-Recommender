# 📈 Automated Stock Recommender

A smart, semi-automated dashboard that provides **daily stock recommendations** based on technical, fundamental, and macroeconomic analysis — powered by AI scoring.

---

## 🚀 Features

### ✅ Core Analysis

- Real-time data from Yahoo Finance (`yfinance`)
- Automatic computation of **RSI, MACD, Moving Averages, ATR, Volume Ratio**
- Basic **fundamental metrics** (P/E, P/B, ROE, DER, Dividend Yield)
- Integrated **AI scoring system** to combine all metrics into actionable recommendations

### 🧠 Recommendation Engine

| Score Range | Signal         | Meaning            |
| ----------- | -------------- | ------------------ |
| 80–100      | 🟢 STRONG BUY  | Very bullish setup |
| 60–79       | 🟡 BUY         | Good upward bias   |
| 40–59       | ⚪ HOLD        | Neutral/Sideway    |
| 20–39       | 🟠 SELL        | Weak setup         |
| 0–19        | 🔴 STRONG SELL | Bearish trend      |

### 🧩 NEW FEATURE: Support–Resistance Timing (2025 Update)

Added visualization and advice section to assist traders in timing entries/exits based on **nearest support and resistance levels.**

- 🔹 **Chart Overlay:** Green dotted line (Support), Red dotted line (Resistance)
- ⚡ **Timing Advice Panel:** Shows nearest levels and AI-generated timing suggestion
- 📊 **AI Summary:** Now includes SR info and actionable commentary

### 🖥️ Streamlit Dashboard

- Dual mode: `Full Auto` (analyze all tickers) or `Manual` (select specific stocks)
- Expandable panels for each stock
- Candlestick chart + MAs + indicators
- Fundamental snapshot
- 3-day trading plan (entry, stop, target)
- Support–Resistance Timing panel
- AI Summary section

### 📂 Extra Tools

- Watchlist (add/remove + export CSV)
- Export full results as CSV
- Auto-refresh option
- Macro context summary (IDX, USDIDR, sector sentiment)

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run gui_full.py
```

---

## 📊 Output Example

Each stock analysis includes:

- Technical summary
- Fundamental quick view
- Support/Resistance zones with chart markers
- AI-based entry/exit suggestion
- 3-day plan

---

## 🧠 Notes

This system is **for decision support and education only**.  
Yahoo Finance data may experience slight delays or missing intervals.

---

### 🪄 2025 Changelog

- Added Support–Resistance detection and timing module
- Enhanced AI Summary integration
- Minor dark-theme visual polish
- Improved chart interaction

---

Created with ❤️ by `panda`
