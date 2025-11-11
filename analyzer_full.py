import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import time
import math
import streamlit as st
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from discord_notifier import DiscordNotifier
import threading

# --- basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoStockAnalyzer")


# Helper
def safe_div(a, b, eps=1e-9):
    return a / (b + eps)


class AutomatedStockAnalyzer:
    """
    AutomatedStockAnalyzer:
    - fetches data via yfinance (with caching)
    - computes technical & deep indicators
    - scores stocks, produces 3-day plan
    - generates intraday buy/sell timing signals
    - simple intraday backtester
    """

    def __init__(self, risk_free_rate: float = 0.05, cache_ttl: float = 300.0):
        self.risk_free_rate = risk_free_rate
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl  # seconds

        # Notif to discord
        self.discord = DiscordNotifier()
        self.whitelist = []
        self._last_signal: Dict[str, str] = {}
        self._notify_lock = threading.Lock()

    def set_watchlist(self, watchlist: list[str]):
        """Sinkronisasi watchlist dari GUI."""
        if not isinstance(watchlist, list):
            watchlist = []
        self.whitelist = watchlist

    def set_discord_notifier(self, webhook_url: str, whitelist: list[str]):
        """Aktifkan notifikasi Discord hanya untuk saham dalam whitelist."""
        try:
            self.discord = DiscordNotifier(webhook_url)
            self.whitelist = whitelist
        except Exception as e:
            print(f"[ERROR] Gagal inisialisasi Discord Notifier: {e}")

    # -------------------------
    # Caching helpers
    # -------------------------
    def _cache_get(self, key: str):
        rec = self.cache.get(key)
        if not rec:
            return None
        if time.time() - rec.get("_ts", 0) > self.cache_ttl:
            self.cache.pop(key, None)
            return None
        return rec.get("val")

    def _cache_set(self, key: str, val: Any):
        self.cache[key] = {"_ts": time.time(), "val": val}

    def clear_cache(self):
        self.cache.clear()

    # -------------------------
    # Data fetching
    # -------------------------
    def get_stock_data(self, ticker: str, period: str, interval: str ) -> Optional[Dict[str, Any]]:
        """
        Returns dict with keys: ticker, hist (DataFrame), info, financials, balance_sheet, cash_flow
        Caches results for cache_ttl seconds.
        """
        key = f"{ticker}_{period}_{interval}"
        cached = self._cache_get(key)
        if cached:
            return cached

        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
            if hist is None or hist.empty:
                logger.warning(f"[get_stock_data] no history for {ticker}")
                return None

            info = {}
            financials = pd.DataFrame()
            balance = pd.DataFrame()
            cashflow = pd.DataFrame()
            try:
                # some tickers may raise on info for certain assets; catch safely
                info = tk.info or {}
                financials = tk.financials
                balance = tk.balance_sheet
                cashflow = tk.cash_flow
            except Exception:
                # non-fatal
                pass

            res = {
                "ticker": ticker,
                "hist": hist,
                "info": info,
                "financials": financials,
                "balance_sheet": balance,
                "cash_flow": cashflow,
            }
            self._cache_set(key, res)
            return res
        except Exception as e:
            logger.exception(f"[get_stock_data] Error {ticker}: {e}")
            return None

    # -------------------------
    # Technical indicators (base)
    # -------------------------
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure columns exist
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan

        # Moving averages
        df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
        df["MA_20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["MA_50"] = df["Close"].rolling(50, min_periods=1).mean()

        # RSI 14 (Wilder's smoothing approximated by EWMA for stability)
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = safe_div(roll_up, roll_down)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # Bollinger Bands
        df["BB_mid"] = df["Close"].rolling(20, min_periods=1).mean()
        df["BB_std"] = df["Close"].rolling(20, min_periods=1).std().fillna(0)
        df["BB_up"] = df["BB_mid"] + 2 * df["BB_std"]
        df["BB_low"] = df["BB_mid"] - 2 * df["BB_std"]

        # Volume moving average & ratio
        df["Vol_MA_20"] = df["Volume"].rolling(20, min_periods=1).mean()
        df["Vol_Ratio"] = safe_div(df["Volume"], df["Vol_MA_20"])

        # ROC (Rate of Change)
        df["ROC_10"] = safe_div((df["Close"] - df["Close"].shift(10)), df["Close"].shift(10)) * 100

        # Support/Resistance simple (20)
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

        # TP and VWAP (rolling approximation)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["TP"] = tp
        window_vwap = 20
        df["VWAP"] = (tp * df["Volume"]).rolling(window=window_vwap, min_periods=1).sum() / (
            df["Volume"].rolling(window=window_vwap, min_periods=1).sum() + 1e-9
        )

        # ATR 14 (True Range)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14, min_periods=1).mean()

        # OBV vectorized (faster)
        close_diff = df["Close"].diff()
        sign = np.sign(close_diff).fillna(0)
        df["OBV"] = (sign * df["Volume"]).fillna(0).cumsum()

        # ADX (vectorized approx)
        up = df["High"].diff()
        down = -df["Low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr = tr.rolling(14, min_periods=1).mean().replace(0, np.nan).fillna(method="ffill").fillna(1e-9)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14, min_periods=1).sum() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14, min_periods=1).sum() / atr)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        df["ADX_14"] = dx.rolling(14, min_periods=1).mean().fillna(0)

        # EMA cross flags for quick checks
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

        net_income = 0
        revenue = 0
        net_income_growth = 0
        revenue_growth = 0
        try:
            if not fin.empty:
                # Note: yfinance financials may have non-standard labels; attempt common ones
                for label in ["Net Income", "Net income", "NetIncome"]:
                    if label in fin.index:
                        net_income = fin.loc[label].iloc[0]
                        break
                for label in ["Total Revenue", "Total revenue", "Revenue"]:
                    if label in fin.index:
                        revenue = fin.loc[label].iloc[0]
                        break
                if fin.shape[1] > 1:
                    # previous column
                    try:
                        prev_net = fin.loc[label].iloc[1]
                        net_income_growth = safe_div(net_income - prev_net, abs(prev_net)) * 100 if prev_net != 0 else 0
                    except Exception:
                        pass
        except Exception:
            pass

        return {
            "current_price": float(current_price) if pd.notna(current_price) else np.nan,
            "market_cap": market_cap,
            "pe_ratio": float(pe) if pe is not None else 0,
            "pb_ratio": float(pb) if pb is not None else 0,
            "eps": trailing_eps,
            "roe": float(roe) * 100 if isinstance(roe, (float, int)) and abs(roe) < 5 else float(roe) if isinstance(roe, (float, int)) else 0,
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
        macros = {}
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
        if df is None or df.empty:
            return {"score": 0, "components": {}, "signal": "NO_DATA"}

        latest = df.iloc[-1]
        # RSI: ideal ~50
        rsi = latest.get("RSI_14", 50)
        rsi_score = max(0, 100 - abs(50 - rsi) * 2)

        # MACD: histogram
        macdh = latest.get("MACD_Hist", 0)
        macd_score = np.clip(50 + (macdh * 10), 0, 100)

        # MA trend
        ema8 = latest.get("EMA_8", 0)
        ema21 = latest.get("EMA_21", 0)
        ema50 = latest.get("EMA_50", 0)
        if ema8 > ema21 > ema50:
            ma_trend_score = 100
        elif ema8 < ema21 < ema50:
            ma_trend_score = 0
        else:
            ma_trend_score = 50

        # Volume surge
        vol_ratio = latest.get("Vol_Ratio", 1)
        vol_score = np.clip((vol_ratio - 1) * 50 + 50, 0, 100)

        total = (rsi_score * 0.2) + (macd_score * 0.3) + (ma_trend_score * 0.3) + (vol_score * 0.2)
        score = float(max(0, min(100, total)))

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

        return {"score": score, "components": components, "signal": signal}

    # -------------------------
    # 3-day decision planner
    # -------------------------
    def generate_3day_decision(
        self,
        df: pd.DataFrame,
        fm: Dict[str, Any],
        capital: float = 100_000_000,
        risk_percent: float = 1.0
    ) -> Dict[str, Any]:
        """
        Optimized 3-day trading plan:
        - Adaptive stop-loss (ATR + support)
        - Adaptive target profit (ATR + resistance + momentum)
        - Position sizing based on risk percent
        """
        if df is None or df.empty:
            return {"action": "NO_DATA"}

        latest = df.iloc[-1]
        entry = float(latest["Close"])
        atr = float(latest.get("ATR_14", 0) or 0)

        # --- Support/Resistance adaptive protection ---
        sup, res = self.detect_support_resistance(df, window=10, sensitivity=0.03)

        # ATR fallback if missing
        if atr <= 0:
            atr = max(1.0, entry * 0.005)

        # --- Adaptive Stop-Loss Calculation ---
        # Base stop by volatility
        base_stop = entry - 1.5 * atr

        # Jika ada support, gunakan yang lebih ketat (lebih realistis)
        if sup and sup < entry:
            adaptive_stop = max(sup * 0.98, base_stop)
        else:
            adaptive_stop = base_stop

        # --- Adaptive Take-Profit Calculation ---
        # Dasar target dari rasio risk:reward
        R = 2.2  # sedikit lebih tinggi dari R=2
        base_target = entry + R * (entry - adaptive_stop)

        # Jika ada resistance, gunakan yang lebih konservatif
        if res and res > entry:
            adaptive_target = min(res * 0.995, base_target)
        else:
            adaptive_target = base_target

        # --- Momentum Adjustment ---
        # Jika momentum kuat (EMA8 > EMA21 > EMA50), naikkan target 10%
        if latest.get("EMA_8", 0) > latest.get("EMA_21", 0) > latest.get("EMA_50", 0):
            adaptive_target *= 1.10
        # Jika momentum melemah, turunkan target sedikit
        elif latest.get("EMA_8", 0) < latest.get("EMA_21", 0):
            adaptive_target *= 0.95

        # --- Position Sizing ---
        risk_amount = capital * (risk_percent / 100.0)
        risk_per_share = entry - adaptive_stop
        if risk_per_share <= 0:
            position_size = 0
        else:
            position_size = int(risk_amount / risk_per_share)
        position_value = position_size * entry

        # --- Action Decision ---
        score_pack = self.score_stock(df, fm)
        score = score_pack["score"]
        roc = latest.get("ROC_10", 0)
        adx = latest.get("ADX_14", 0)

        action = "HOLD"
        if score >= 80 and adx > 25:
            action = "ENTER_LONG"
        elif score >= 60 and roc > 0:
            action = "ENTER_LONG"
        elif score < 35 or adx < 15:
            action = "EXIT_OR_SHORT"

        return {
            "action": action,
            "entry": round(entry, 6),
            "stop": round(adaptive_stop, 6),
            "target": round(adaptive_target, 6),
            "position_size": int(position_size),
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "score": round(score, 2),
        }


    # -------------------------
    # Support–Resistance detection
    # -------------------------
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 10, sensitivity: float = 0.03) -> Tuple[Optional[float], Optional[float]]:
        """
        Simpler peak/trough based S/R detection using rolling center window.
        Returns (nearest_support, nearest_resistance) relative to last price within sensitivity.
        """
        if df is None or df.empty:
            return None, None
        df = df.copy()
        # Centered rolling min/max
        df["center_min"] = df["Low"].rolling(window=window, center=True, min_periods=1).min()
        df["center_max"] = df["High"].rolling(window=window, center=True, min_periods=1).max()

        unique_supports = sorted(df["center_min"].dropna().unique())
        unique_resistances = sorted(df["center_max"].dropna().unique())

        last_price = df["Close"].iloc[-1]
        nearby_supports = [s for s in unique_supports if s <= last_price * (1 + sensitivity)]
        nearby_resistances = [r for r in unique_resistances if r >= last_price * (1 - sensitivity)]

        nearest_support = max(nearby_supports) if nearby_supports else None
        nearest_resistance = min(nearby_resistances) if nearby_resistances else None

        return nearest_support, nearest_resistance

    # -------------------------
    # Timing advice generator
    # -------------------------
    def timing_signal(self, last_price: float, nearest_support: Optional[float], nearest_resistance: Optional[float], main_signal: str) -> str:
        if nearest_support and last_price <= nearest_support * 1.01 and main_signal in ["BUY", "HOLD", "STRONG BUY"]:
            return f"🟢 Potensi Entry di sekitar support {nearest_support}"
        elif nearest_resistance and last_price >= nearest_resistance * 0.99 and main_signal in ["SELL", "HOLD"]:
            return f"🔴 Potensi Take Profit di sekitar resistance {nearest_resistance}"
        elif nearest_resistance and last_price > nearest_resistance:
            return f"🚀 Breakout di atas {nearest_resistance}"
        elif nearest_support and last_price < nearest_support:
            return f"⚠️ Breakdown di bawah {nearest_support}"
        else:
            return "⏸ Tidak ada sinyal timing signifikan saat ini."

    # -------------------------
    # Intraday signal generator
    # -------------------------
    def generate_intraday_signals(
        self,
        df: pd.DataFrame,
        lookback_ema_fast: int = 8,
        lookback_ema_slow: int = 21,
        macd_hist_threshold: float = 0.0,
        rsi_buy_thresh: float = 40,
        rsi_sell_thresh: float = 60,
        vol_spike_multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """
        Returns DataFrame with timestamp index and columns: signal (BUY/SELL/HOLD), reason, confidence [0..1].
        Combines: EMA cross, MACD histogram turning positive/negative, VWAP cross, RSI, Volume spike.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy().sort_index()
        # Ensure deep indicators exist
        if f"EMA_{lookback_ema_fast}" not in df.columns or f"EMA_{lookback_ema_slow}" not in df.columns:
            df = self.calculate_deep_technical_indicators(df)

        signals = []
        prev_row = None
        for idx, row in df.iterrows():
            reasons = []
            confidence = 0.0

            # EMA cross (fast crossing above slow)
            ema_fast = row.get(f"EMA_{lookback_ema_fast}", np.nan)
            ema_slow = row.get(f"EMA_{lookback_ema_slow}", np.nan)
            if prev_row is not None:
                prev_fast = prev_row.get(f"EMA_{lookback_ema_fast}", np.nan)
                prev_slow = prev_row.get(f"EMA_{lookback_ema_slow}", np.nan)
                # Golden cross event
                if prev_fast <= prev_slow and ema_fast > ema_slow:
                    reasons.append("EMA_CROSS_UP")
                    confidence += 0.25
                elif prev_fast >= prev_slow and ema_fast < ema_slow:
                    reasons.append("EMA_CROSS_DOWN")
                    confidence += 0.25

            # MACD hist turning
            macd_hist = row.get("MACD_Hist", 0)
            if prev_row is not None:
                prev_macd = prev_row.get("MACD_Hist", 0)
                if prev_macd <= macd_hist_threshold and macd_hist > macd_hist_threshold:
                    reasons.append("MACD_TURN_POS")
                    confidence += 0.2
                elif prev_macd >= macd_hist_threshold and macd_hist < macd_hist_threshold:
                    reasons.append("MACD_TURN_NEG")
                    confidence += 0.2

            # VWAP cross
            vwap = row.get("VWAP", np.nan)
            close = row.get("Close", np.nan)
            if not np.isnan(vwap):
                if prev_row is not None:
                    prev_close = prev_row.get("Close", np.nan)
                    prev_vwap = prev_row.get("VWAP", np.nan)
                    if prev_close <= prev_vwap and close > vwap:
                        reasons.append("VWAP_CROSS_UP")
                        confidence += 0.15
                    elif prev_close >= prev_vwap and close < vwap:
                        reasons.append("VWAP_CROSS_DOWN")
                        confidence += 0.15

            # RSI bias
            rsi = row.get("RSI_14", 50)
            if rsi < rsi_buy_thresh:
                reasons.append("RSI_OVERSOLD")
                confidence += 0.05
            elif rsi > rsi_sell_thresh:
                reasons.append("RSI_OVERBOUGHT")
                confidence += 0.05

            # Volume spike
            vol_ratio = row.get("Vol_Ratio", 1)
            if vol_ratio >= vol_spike_multiplier:
                reasons.append("VOL_SPIKE")
                confidence += 0.1

            # Normalize confidence to 0..1
            confidence = float(min(1.0, confidence))

            # Decision mapping (simple rule-based)
            signal = "HOLD"
            if any(r in ("EMA_CROSS_UP", "MACD_TURN_POS", "VWAP_CROSS_UP") for r in reasons) and confidence >= 0.25:
                signal = "BUY"
            if any(r in ("EMA_CROSS_DOWN", "MACD_TURN_NEG", "VWAP_CROSS_DOWN") for r in reasons) and confidence >= 0.25:
                signal = "SELL"

            signals.append({"timestamp": idx, "signal": signal, "reasons": ",".join(reasons) if reasons else "", "confidence": confidence})
            prev_row = row

        return pd.DataFrame(signals).set_index("timestamp")

    # -------------------------
    # Simplified intraday backtester
    # -------------------------
    def backtest_intraday(self, df: pd.DataFrame, signals: pd.DataFrame, slippage: float = 0.0005, fee: float = 0.0005) -> Dict[str, Any]:
        """
        Simulate simple strategy:
        - Enter at close price on BUY signal, exit on SELL or target/stop (not implemented complex TP/SL here).
        - Returns P&L summary and trades list.
        This is simple but useful to test signal logic on historical intraday data.
        """
        trades = []
        position = None
        for idx, s in signals.iterrows():
            price_row = df.loc[idx] if idx in df.index else None
            if price_row is None:
                continue
            price = float(price_row["Close"])
            if s["signal"] == "BUY" and position is None:
                entry_price = price * (1 + slippage + fee)
                position = {"entry_time": idx, "entry_price": entry_price, "confidence": s["confidence"], "reasons": s["reasons"]}
            elif s["signal"] == "SELL" and position is not None:
                exit_price = price * (1 - slippage - fee)
                pnl = exit_price - position["entry_price"]
                trades.append({**position, "exit_time": idx, "exit_price": exit_price, "pnl": pnl})
                position = None

        # If still open, close at last available price
        if position is not None:
            last_price = float(df["Close"].iloc[-1])
            exit_price = last_price * (1 - slippage - fee)
            pnl = exit_price - position["entry_price"]
            trades.append({**position, "exit_time": df.index[-1], "exit_price": exit_price, "pnl": pnl})

        total_pnl = sum(t["pnl"] for t in trades) if trades else 0.0
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        return {"trades": trades, "total_pnl": total_pnl, "wins": wins, "losses": losses, "num_trades": len(trades)}

    # -------------------------
    # Single stock full analysis pipeline (final unified)
    # -------------------------
    def analyze_one(self, ticker: str, period: str , interval: str, capital: float, risk_percent: float) -> Optional[Dict[str, Any]]:
        stock_data = self.get_stock_data(ticker, period=period, interval=interval)
        if not stock_data:
            return None

        df = stock_data["hist"].copy()
        # Ensure index is datetime sorted
        df = df.sort_index()

        df = self.calculate_technical_indicators(df)
        df = self.calculate_deep_technical_indicators(df)
        fm = self.calculate_fundamental_metrics(stock_data)
        score_pack = self.score_stock(df, fm)
        plan = self.generate_3day_decision(df, fm, capital=capital, risk_percent=risk_percent)

        # Support/Resistance + timing
        support, resistance = self.detect_support_resistance(df)
        timing = self.timing_signal(float(df["Close"].iloc[-1]), support, resistance, score_pack["signal"])

        # Intraday signals and backtest
        intraday_signals = self.generate_intraday_signals(df)
        backtest_summary = self.backtest_intraday(df, intraday_signals)

        out = {
            "ticker": ticker,
            "price_data": df,
            "fundamental_metrics": fm,
            "technical_score": score_pack,
            "3day_plan": plan,
            "support": support,
            "resistance": resistance,
            "timing_advice": timing,
            "intraday_signals": intraday_signals,
            "intraday_backtest": backtest_summary,
        }

        # --- Discord Notification (Embed Version) ---
        try:
            main_signal = score_pack.get("signal", "")
            action = plan.get("action", "")
            price = float(df["Close"].iloc[-1])
            confidence = float(score_pack.get("score", 0)) / 100
            summary = timing

            # Hanya kirim jika ticker termasuk dalam whitelist
            if ticker in self.whitelist:
                # Tentukan tipe sinyal untuk tracking perubahan
                alert_signal = None
                if main_signal == "STRONG BUY":
                    alert_signal = "STRONG_BUY"
                elif action == "ENTER_LONG":
                    alert_signal = "BUY"
                elif action == "EXIT_OR_SHORT":
                    alert_signal = "SELL"

                if alert_signal:
                    with self._notify_lock:
                        prev_signal = self._last_signal.get(ticker)
                        # hanya kirim jika sinyal berubah
                        if prev_signal != alert_signal:
                            self._last_signal[ticker] = alert_signal
                            should_send = True
                        else:
                            should_send = False

                    if should_send:
                        # Tentukan tampilan embed
                        embed_title = None
                        embed_desc = None
                        color = 0x808080

                        if alert_signal == "STRONG_BUY":
                            color = 0x16A34A
                            embed_title = f"🚀 STRONG BUY ALERT — {ticker}"
                            embed_desc = (
                                f"**Score:** {score_pack['score']:.2f}\n"
                                f"**Entry:** {plan['entry']}\n"
                                f"**Target:** {plan['target']}\n"
                                f"**Stop:** {plan['stop']}\n\n"
                                f"**Timing:** {timing}"
                            )
                        elif alert_signal == "BUY":
                            color = 0x22C55E
                            embed_title = f"🟢 BUY SIGNAL — {ticker}"
                            embed_desc = (
                                f"**Entry:** {plan['entry']} | **Target:** {plan['target']} | **Stop:** {plan['stop']}\n"
                                f"Technical Score: {score_pack['score']:.1f}\n\n"
                                f"{timing}"
                            )
                        elif alert_signal == "SELL":
                            color = 0xDC2626
                            embed_title = f"🔴 SELL/EXIT ALERT — {ticker}"
                            embed_desc = (
                                f"**Close price:** {plan['entry']}\n"
                                f"**Signal:** {main_signal}\n\n"
                                f"{timing}"
                            )

                        # Kirim embed ke Discord (pakai cooldown internal)
                        if embed_title:
                            self.discord.send_embed(
                                embed_title,
                                embed_desc,
                                color,
                                ticker=ticker,
                                signal=alert_signal
                            )
                            logger.info(f"📢 Discord alert sent for {ticker}: {alert_signal}")

                        # Kirim juga pesan teks ringkas (opsional)
                        msg = f"{alert_signal} signal detected for {ticker} at {price:.2f}"
                        self.discord.send_message(msg, ticker=ticker, signal=alert_signal)

        except Exception as e:
            logger.error(f"[DiscordNotifier] Gagal kirim notifikasi untuk {ticker}: {e}")
        # --- End Discord Notification ---

        return out

    # -------------------------
    # Batch runner to get top N recommendations (improved)
    # -------------------------
    def generate_recommendations(
        self,
        tickers: List[str],
        period: str,
        interval: str,
        top_n: int,
        risk_percent: float,
        capital: float = 1_000_000,
    ) -> Dict[str, Any]:
        results = {}
        ranked = []

        BATCH_SIZE = 100
        ticker_batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
        total_batches = len(ticker_batches)

        # streamlit progress (optional)
        try:
            progress_bar = st.progress(0, text="⏳ Starting stock analysis...")
            progress_text = st.empty()
        except Exception:
            progress_bar = None
            progress_text = None

        start_time = time.time()

        def process_batch(batch_index: int, batch: List[str]):
            batch_results = []
            for t in batch:
                try:
                    analysis = self.analyze_one(t, period=period, interval=interval, capital=capital, risk_percent=risk_percent)
                    if analysis:
                        score = analysis["technical_score"]["score"]
                        results[t] = analysis
                        batch_results.append((t, score))
                    time.sleep(random.uniform(0.1, 0.45))  # tiny delay to reduce throttling
                except Exception as e:
                    logger.exception(f"[generate_recommendations] {t} error: {e}")
            return batch_results

        all_results = []
        max_workers = min(6, max(1, math.ceil(total_batches / 2)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_batch, idx + 1, batch): idx + 1 for idx, batch in enumerate(ticker_batches)}
            completed = 0
            for future in as_completed(future_to_batch):
                completed += 1
                try:
                    batch_res = future.result()
                    all_results.extend(batch_res)
                except Exception as e:
                    logger.exception(f"[generate_recommendations] batch error: {e}")

                # update streamlit progress optionally
                if progress_bar and progress_text:
                    elapsed = time.time() - start_time
                    percent = int((completed / total_batches) * 100)
                    progress_bar.progress(percent, text=f"🚀 Processing batch {completed}/{total_batches} ({percent}%)")
                    progress_text.text(f"✅ Batch {completed}/{total_batches} selesai — collected {len(all_results)} scores")
                time.sleep(0.5)

        # finalize progress
        if progress_bar:
            progress_bar.progress(100, text="✅ All analysis complete!")
            progress_text.text("🎉 Semua analisis selesai.")

        ranked_sorted = sorted(all_results, key=lambda x: x[1], reverse=True)
        top = ranked_sorted[:top_n]
        macro = self.analyze_macro_context()
        return {"results": results, "ranked": ranked_sorted, "top": top, "macro": macro}
    
        



    # -------------------------
    # Long-term investment simulator (Monte Carlo + analyst scenarios)
    # -------------------------
    def historical_cagr(self, df: pd.DataFrame, years: float) -> float:
        """Calculate historical CAGR from df['Close'] annualized over available history."""
        if df is None or df.empty:
            return 0.0
        prices = df["Close"].dropna()
        if prices.empty:
            return 0.0
        n_days = (prices.index[-1] - prices.index[0]).days or 1
        total_years = n_days / 365.25
        start = float(prices.iloc[0])
        end = float(prices.iloc[-1])
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / total_years) - 1.0
        return float(cagr)

    def annualized_vol_from_hist(self, df: pd.DataFrame) -> float:
        """Estimate annualized volatility from historical daily returns."""
        if df is None or df.empty:
            return 0.0
        rets = df["Close"].pct_change().dropna()
        if rets.empty:
            return 0.0
        daily_std = float(rets.std())
        annual_vol = daily_std * (252 ** 0.5)
        return float(annual_vol)

    def simulate_long_term_investment(
        self,
        ticker: str,
        years: float ,
        initial_capital: float ,
        periodic_contribution: float ,
        contribution_freq_per_year: int,
        n_simulations: int,
        method: str,
        analyst_estimates: Optional[list] = None,
        use_bootstrap: bool = True,
        seed: Optional[int] = None,
    ) -> dict:
            """
            Powerfull Long-Term Investment Simulator:
            - Monte Carlo (bootstrap or GBM hybrid)
            - Deterministic (historical CAGR)
            - Analyst scenarios (multi-weighted)
            - Sensitivity analysis (±return, ±volatility)
            """

            # --- Fetch historical data robustly ---
            try:
                stock_data = self.get_stock_data(ticker, period="max", interval="1d") or {}
                df = stock_data.get("hist", None)
            except Exception as e:
                logger.error(f"[simulate_long_term_investment] data fetch error: {e}")
                df = None

            if df is None or df.empty:
                return {"error": "no_hist_data", "ticker": ticker}

            df = df.sort_index().asfreq("B", method="ffill")
            daily_rets = df["Close"].pct_change().dropna()

            if daily_rets.empty:
                return {"error": "no_returns", "ticker": ticker}

            # --- Historical stats ---
            mu_daily = float(daily_rets.mean())
            sigma_daily = float(daily_rets.std())
            annualized_mu = (1 + mu_daily) ** 252 - 1
            annualized_vol = sigma_daily * (252 ** 0.5)
            hist_cagr = self.historical_cagr(df, years)

            rng = np.random.RandomState(seed or int(time.time()))
            steps_per_year = contribution_freq_per_year
            total_periods = int(round(years * steps_per_year))
            dt = 1.0 / steps_per_year

            # --- Analyst estimates weighted ---
            analyst_results = []
            weighted_mu = annualized_mu
            weighted_vol = annualized_vol
            if analyst_estimates:
                total_w = 0
                agg_mu = 0
                for a in analyst_estimates:
                    if isinstance(a, dict):
                        r = a.get("r", 0.07)
                        w = a.get("w", 1.0)
                    else:
                        r = float(a)
                        w = 1.0
                    total_w += w
                    agg_mu += r * w
                    final = initial_capital * ((1 + r) ** years)
                    if periodic_contribution:
                        m = contribution_freq_per_year
                        rate = r / m
                        fv_contrib = periodic_contribution * (((1 + rate) ** (m * years) - 1) / rate) if rate != 0 else periodic_contribution * m * years
                        final += fv_contrib
                    analyst_results.append({"annual_return": r, "weight": w, "final_value": final})
                weighted_mu = agg_mu / max(1e-9, total_w)

            # --- Deterministic branch ---
            if method == "deterministic":
                hist_proj = initial_capital * ((1 + hist_cagr) ** years)
                if periodic_contribution:
                    m = contribution_freq_per_year
                    rate = hist_cagr / m if hist_cagr != 0 else 0
                    fv_contrib = periodic_contribution * (((1 + rate) ** (m * years) - 1) / rate) if rate != 0 else periodic_contribution * m * years
                    hist_proj += fv_contrib
                return {
                    "method": "deterministic",
                    "ticker": ticker,
                    "years": years,
                    "initial_capital": initial_capital,
                    "periodic_contribution": periodic_contribution,
                    "historical_cagr": hist_cagr,
                    "projection": hist_proj,
                    "analyst_scenarios": analyst_results,
                    "annualized_mu": annualized_mu,
                    "annualized_vol": annualized_vol,
                }

            # --- Monte Carlo hybrid ---
            simulations_final = []
            gbm_paths = [] if n_simulations <= 500 else None
            for sim in range(n_simulations):
                val = initial_capital
                path = [val]
                last_price = df["Close"].iloc[-1]
                for _ in range(total_periods):
                    # hybrid: combine GBM + bootstrap noise
                    if use_bootstrap and random.random() < 0.5:
                        # bootstrap return
                        r = rng.choice(daily_rets.values, size=int(252 / steps_per_year)).mean() * steps_per_year
                    else:
                        # GBM random
                        drift = (weighted_mu - 0.5 * weighted_vol ** 2) * dt
                        shock = weighted_vol * math.sqrt(dt) * rng.normal()
                        r = drift + shock
                    val = val * (1 + r)
                    if periodic_contribution:
                        val += periodic_contribution
                    path.append(val)
                simulations_final.append(val)
                if gbm_paths is not None:
                    gbm_paths.append(path)

            arr = np.array(simulations_final)
            percentiles = {p: float(np.percentile(arr, p)) for p in [5, 25, 50, 75, 95]}
            mean = float(arr.mean())
            std = float(arr.std())
            prob_loss = float(np.mean(arr < initial_capital))
            exp_cagr = (mean / initial_capital) ** (1 / years) - 1
            sharpe = safe_div((annualized_mu - self.risk_free_rate), annualized_vol)

            # --- Sensitivity analysis ---
            sensitivity = {
                "+20% Return": mean * 1.2,
                "-20% Return": mean * 0.8,
                "+10% Vol": std * 1.1,
                "-10% Vol": std * 0.9,
            }

            return {
                "method": "mc",
                "ticker": ticker,
                "years": years,
                "initial_capital": initial_capital,
                "periodic_contribution": periodic_contribution,
                "n_simulations": n_simulations,
                "annualized_mu": annualized_mu,
                "annualized_vol": annualized_vol,
                "weighted_mu": weighted_mu,
                "mc_percentiles": percentiles,
                "mc_mean": mean,
                "mc_std": std,
                "prob_loss": prob_loss,
                "expected_cagr": exp_cagr,
                "sharpe_ratio": sharpe,
                "sensitivity": sensitivity,
                "analyst_scenarios": analyst_results,
                "sample_paths": gbm_paths,
            }


