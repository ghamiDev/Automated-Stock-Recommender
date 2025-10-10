import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Helper
def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

class AutomatedStockAnalyzer:
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.cache = {}

    # -------------------------
    # Data fetching
    # -------------------------
    def get_stock_data(self, ticker: str, period: str = "1mo", interval: str = "60m") -> Dict[str, Any]:
        """
        Fetch historical price data and (if available) fundamentals via yfinance.
        period: e.g. "1mo", "3mo", "6mo", "1y"
        interval: e.g. "1m","5m","60m","1d" - yfinance supports certain combos
        Returns dict: {'hist': DataFrame, 'info': dict, 'financials': df...}
        """
        key = f"{ticker}_{period}_{interval}"
        if key in self.cache:
            return self.cache[key]

        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
            if hist is None or hist.empty:
                return None

            info = {}
            financials = pd.DataFrame()
            balance = pd.DataFrame()
            cashflow = pd.DataFrame()
            try:
                info = tk.info or {}
                financials = tk.financials
                balance = tk.balance_sheet
                cashflow = tk.cash_flow
            except Exception:
                # some tickers may not provide full fundamentals
                pass

            res = {
                "ticker": ticker,
                "hist": hist,
                "info": info,
                "financials": financials,
                "balance_sheet": balance,
                "cash_flow": cashflow,
            }
            self.cache[key] = res
            return res
        except Exception as e:
            print(f"[get_stock_data] Error {ticker}: {e}")
            return None

    # -------------------------
    # Technical indicators (base)
    # -------------------------
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ensure numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        # MAs
        df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
        df["MA_20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["MA_50"] = df["Close"].rolling(50, min_periods=1).mean()
        # RSI 14
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(14, min_periods=1).mean()
        roll_down = down.rolling(14, min_periods=1).mean()
        rs = safe_div(roll_up, roll_down)
        df["RSI_14"] = 100 - (100 / (1 + rs))
        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        # Bollinger
        df["BB_mid"] = df["Close"].rolling(20, min_periods=1).mean()
        df["BB_std"] = df["Close"].rolling(20, min_periods=1).std().fillna(0)
        df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]
        df["BB_low"] = df["BB_mid"] - 2 * df["BB_std"]
        # Volume MA
        df["Vol_MA_20"] = df["Volume"].rolling(20, min_periods=1).mean()
        df["Vol_Ratio"] = safe_div(df["Volume"], df["Vol_MA_20"])
        # ROC
        df["ROC_10"] = safe_div((df["Close"] - df["Close"].shift(10)), df["Close"].shift(10)) * 100
        # Support/Resistance simple
        df["Res_20"] = df["High"].rolling(20, min_periods=1).max()
        df["Sup_20"] = df["Low"].rolling(20, min_periods=1).min()
        return df

    # -------------------------
    # Deep technical indicators
    # -------------------------
    def calculate_deep_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # EMAs
        for s in (8, 21, 50, 100, 200):
            df[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
        # VWAP (rolling window approx)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["TP"] = tp
        window_vwap = 20
        df["VWAP"] = (tp * df["Volume"]).rolling(window=window_vwap, min_periods=1).sum() / (df["Volume"].rolling(window=window_vwap, min_periods=1).sum() + 1e-9)
        # ATR 14
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14, min_periods=1).mean()
        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iat[i] > df["Close"].iat[i - 1]:
                obv.append(obv[-1] + df["Volume"].iat[i])
            elif df["Close"].iat[i] < df["Close"].iat[i - 1]:
                obv.append(obv[-1] - df["Volume"].iat[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv
        # ADX simple approx
        up = df["High"].diff()
        down = -df["Low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr = tr.rolling(14, min_periods=1).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(14, min_periods=1).sum() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm).rolling(14, min_periods=1).sum() / (atr + 1e-9))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        df["ADX_14"] = dx.rolling(14, min_periods=1).mean().fillna(0)
        # EMA cross flags
        df["EMA8_above_EMA21"] = (df["EMA_8"] > df["EMA_21"]).astype(int)
        df["EMA21_above_EMA50"] = (df["EMA_21"] > df["EMA_50"]).astype(int)
        return df

    # -------------------------
    # Fundamental metrics
    # -------------------------
    def calculate_fundamental_metrics(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        info = stock_data.get("info", {}) or {}
        fin = stock_data.get("financials", pd.DataFrame())
        bal = stock_data.get("balance_sheet", pd.DataFrame())
        cash = stock_data.get("cash_flow", pd.DataFrame())

        current_price = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose", np.nan)))
        market_cap = info.get("marketCap", 0)
        trailing_eps = info.get("trailingEps", None) or info.get("eps", None) or 0
        pe = safe_div(current_price, trailing_eps) if trailing_eps else info.get("trailingPE", 0) or 0
        pb = info.get("priceToBook", 0) or safe_div(current_price, info.get("bookValue", 0))
        roe = info.get("returnOnEquity", 0) or 0
        der = info.get("debtToEquity", 0) or 0
        div_yield = info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0

        # attempts to compute growth if financials present
        net_income = 0
        revenue = 0
        net_income_growth = 0
        revenue_growth = 0
        try:
            if not fin.empty and "Net Income" in fin.index:
                net_income = fin.loc["Net Income"].iloc[0]
            if not fin.empty and "Total Revenue" in fin.index:
                revenue = fin.loc["Total Revenue"].iloc[0]
            if not fin.empty and fin.shape[1] > 1:
                prev_net = fin.loc["Net Income"].iloc[1] if "Net Income" in fin.index else net_income
                prev_rev = fin.loc["Total Revenue"].iloc[1] if "Total Revenue" in fin.index else revenue
                net_income_growth = safe_div(net_income - prev_net, abs(prev_net)) * 100 if prev_net != 0 else 0
                revenue_growth = safe_div(revenue - prev_rev, abs(prev_rev)) * 100 if prev_rev != 0 else 0
        except Exception:
            pass

        return {
            "current_price": float(current_price) if pd.notna(current_price) else np.nan,
            "market_cap": market_cap,
            "pe_ratio": float(pe) if pe is not None else 0,
            "pb_ratio": float(pb) if pb is not None else 0,
            "eps": trailing_eps,
            "roe": float(roe) * 100 if isinstance(roe, (float, int)) and abs(roe) < 5 else float(roe) if isinstance(roe, (float, int)) else 0,  # sometimes already percent
            "der": float(der),
            "dividend_yield": float(div_yield),
            "net_income": net_income,
            "revenue": revenue,
            "net_income_growth": net_income_growth,
            "revenue_growth": revenue_growth,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
        }

    # -------------------------
    # Macro context analysis
    # -------------------------
    def analyze_macro_context(self) -> Dict[str, Any]:
        """
        Simple macro context using yfinance tickers:
        - IHSG / JKSE: ^JKSE  (May be limited in yfinance; fallback)
        - USDIDR: USDIDR=X
        - Gold: GC=F
        - Oil: CL=F
        Returns dict with quick signals and values.
        """
        macros = {}
        # symbol list
        symbols = {
            "IHSG": "^JKSE",
            "USDIDR": "USDIDR=X",
            "GOLD": "GC=F",
            "OIL": "CL=F",
        }
        for name, sym in symbols.items():
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(period="5d", interval="1d")
                if hist is None or hist.empty:
                    macros[name] = {"symbol": sym, "error": "no data"}
                    continue
                last = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
                pct = safe_div(last - prev, prev) * 100 if prev != 0 else 0
                macros[name] = {"symbol": sym, "last": float(last), "pct": float(pct)}
            except Exception as e:
                macros[name] = {"symbol": sym, "error": str(e)}

        # quick sentiment derived
        sentiment = "Neutral"
        ihsg_pct = macros.get("IHSG", {}).get("pct", 0)
        usd_pct = macros.get("USDIDR", {}).get("pct", 0)
        if ihsg_pct > 0.2 and usd_pct < 0:
            sentiment = "Risk-on"
        elif ihsg_pct < -0.2 or usd_pct > 0.2:
            sentiment = "Risk-off"

        return {"macros": macros, "market_sentiment": sentiment}

    # -------------------------
    # Scoring / Recommender
    # -------------------------
    def score_stock(self, df: pd.DataFrame, fm: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute composite score:
        Score = (RSI_score*0.2) + (MACD_score*0.3) + (MA_trend*0.3) + (Volume*0.2)
        Return details with component scores.
        Component scales 0-100.
        """
        if df is None or df.empty:
            return {"score": 0, "components": {}, "signal": "NO_DATA"}

        latest = df.iloc[-1]
        # RSI score: best around 50, penalize overbought/oversold
        rsi = latest.get("RSI_14", 50)
        # map [0..100] to score where 50 => 100 (best), extremes worse
        rsi_score = max(0, 100 - abs(50 - rsi) * 2)

        # MACD score: positive histogram -> good
        macdh = latest.get("MACD_Hist", 0)
        macd_score = np.clip(50 + (macdh * 10), 0, 100)  # scaled

        # MA trend score: check EMA8/21/50
        ema8 = latest.get("EMA_8", 0)
        ema21 = latest.get("EMA_21", 0)
        ema50 = latest.get("EMA_50", 0)
        ma_trend_score = 50
        if ema8 > ema21 > ema50:
            ma_trend_score = 100
        elif ema8 < ema21 < ema50:
            ma_trend_score = 0
        else:
            ma_trend_score = 50

        # Volume surge score
        vol_ratio = latest.get("Vol_Ratio", 1)
        vol_score = np.clip((vol_ratio - 1) * 50 + 50, 0, 100)

        # combine
        total = (rsi_score * 0.2) + (macd_score * 0.3) + (ma_trend_score * 0.3) + (vol_score * 0.2)
        score = max(0, min(100, total))

        # signal mapping
        if score >= 80:
            signal = "STRONG BUY"
        elif score >= 60:
            signal = "BUY"
        elif score >= 40:
            signal = "HOLD"
        else:
            signal = "SELL"

        components = {
            "rsi_score": float(rsi_score),
            "macd_score": float(macd_score),
            "ma_trend_score": float(ma_trend_score),
            "volume_score": float(vol_score),
        }

        return {"score": float(score), "components": components, "signal": signal}

    # -------------------------
    # 3-day decision planner
    # -------------------------
    def generate_3day_decision(self, df: pd.DataFrame, fm: Dict[str, Any], capital: float = 100000000, risk_percent: float = 1.0) -> Dict[str, Any]:
        """
        Using ATR-based stop & R=2 target, compute position sizing for 3-day horizon.
        Returns a plan dict.
        """
        if df is None or df.empty:
            return {"action": "NO_DATA"}

        latest = df.iloc[-1]
        entry = float(latest["Close"])
        atr = float(latest.get("ATR_14", 0) or 0)
        if atr <= 0:
            stop = entry * 0.99
            atr_distance = entry - stop
        else:
            stop = entry - 1.5 * atr
            atr_distance = max(0.0001, entry - stop)

        R = 2.0
        target = entry + R * atr_distance

        risk_amount = capital * (risk_percent / 100.0)
        position_size = int(max(0, risk_amount / (atr_distance + 1e-9)))
        position_value = position_size * entry

        # decide action based on score + momentum
        score_pack = self.score_stock(df, fm)
        score = score_pack["score"]
        roc = latest.get("ROC_10", 0)
        action = "HOLD"
        if score >= 80:
            action = "ENTER_LONG"
        elif score >= 60 and roc > 0:
            action = "ENTER_LONG"
        elif score < 35:
            action = "EXIT_OR_SHORT"
        else:
            action = "HOLD"

        plan = {
            "action": action,
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(target, 4),
            "position_size": int(position_size),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "score": round(score, 2),
        }
        return plan

    # -------------------------
    # Single stock full analysis pipeline
    # -------------------------
    def analyze_one(self, ticker: str, period: str = "1mo", interval: str = "60m", capital: float = 100000000, risk_percent: float = 1.0) -> Dict[str, Any]:
        stock_data = self.get_stock_data(ticker, period=period, interval=interval)
        if not stock_data:
            return None

        df = stock_data["hist"].copy()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_deep_technical_indicators(df)
        fm = self.calculate_fundamental_metrics(stock_data)
        score_pack = self.score_stock(df, fm)
        plan = self.generate_3day_decision(df, fm, capital=capital, risk_percent=risk_percent)

        out = {
            "ticker": ticker,
            "price_data": df,
            "fundamental_metrics": fm,
            "technical_score": score_pack,
            "3day_plan": plan,
        }
        return out

    # -------------------------
    # Batch runner to get top N recommendations
    # -------------------------
    def generate_recommendations(self, tickers: List[str], period: str = "1mo", interval: str = "60m", top_n: int = 10, capital: float = 100000000, risk_percent: float = 1.0) -> Dict[str, Any]:
        """
        Analyze provided tickers and return top_n recommended stocks sorted by score.
        Returns dict with:
         - 'results': {ticker: analysis}
         - 'ranked': list of (ticker, score)
         - 'macro': macro_context
        """
        results = {}
        ranked = []
        for t in tickers:
            try:
                analysis = self.analyze_one(t, period=period, interval=interval, capital=capital, risk_percent=risk_percent)
                if analysis:
                    score = analysis["technical_score"]["score"]
                    results[t] = analysis
                    ranked.append((t, score))
            except Exception as e:
                print(f"[generate_recommendations] {t} error: {e}")

        ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)
        top = ranked_sorted[:top_n]
        macro = self.analyze_macro_context()
        return {"results": results, "ranked": ranked_sorted, "top": top, "macro": macro}

    # -------------------------
    # small utility to clear cache
    # -------------------------
    def clear_cache(self):
        self.cache.clear()
