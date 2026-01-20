from __future__ import annotations
from concurrent.futures import as_completed
import os
import json
import time
import math
import random
import tempfile
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import concurrent
from flask import signals
import numpy as np
import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Optional import point: if DiscordNotifier not available, provide noop fallback
try:
    from discord_notifier import DiscordNotifier
except ImportError:
    class DiscordNotifier:
        def __init__(self, webhook: str | None = None):
            self.webhook = webhook
            self.enabled = webhook is not None  # Tambah flag
        
        def send_embed(self, *args, **kwargs):
            if self.enabled:
                # Implementasi fallback atau log
                pass
        def send_message(self, *args, **kwargs): pass
        def send_watchlist_added(self, *args, **kwargs): pass

# Logging
logger = logging.getLogger("AutomatedStockAnalyzer")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------------------------
# Helpers
# ---------------------------
def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    try:
        return a / (b + eps)
    except Exception:
        return 0.0

def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically to avoid corruption with concurrent writers."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def read_json_safe(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # try to repair trivial cases: load line by line first JSON object
        try:
            text = open(path, "r", encoding="utf-8").read()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1 and end > start:
                partial = text[start:end]
                return json.loads(partial)
        except Exception:
            pass
    # fallback: reset file
    try:
        atomic_write_json(path, {})
    except Exception:
        pass
    return {}

# ---------------------------
# Analyzer class
# ---------------------------
class Config:
    DEFAULT_PERIOD = "3mo"
    DEFAULT_INTERVAL = "1d"
    MAX_WORKERS = 6
    CACHE_TTL = 300
    MIN_VOLUME = 500_000
class AutomatedStockAnalyzer:
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        cache_ttl: float = 300.0,
        history_folder: str = "trade_history",
        history_file: str = "signal_history.json",
        enable_pattern_detection: bool = True,
    ):
        self._cache_lock = threading.Lock()
        self.risk_free_rate = float(risk_free_rate)
        self.cache_ttl = float(cache_ttl)
        self._cache: dict[str, dict] = {}
        self._cache_ts: dict[str, float] = {}

        # Discord notifier
        self.discord = DiscordNotifier()
        self.whitelist: list[str] = []
        self._last_signal: dict[str, str] = {}
        self._notify_lock = threading.Lock()
        self._discord_cooldown = {}  # ticker -> timestamp of last send
        self.discord_cooldown_seconds = 300  # default 5 minutes cooldown

        # Winrate/history storage (thread-safe)
        self.history_folder = os.path.abspath(history_folder)
        self.history_file = os.path.abspath(os.path.join(self.history_folder, history_file))
        self._history_lock = threading.Lock()
        os.makedirs(self.history_folder, exist_ok=True)
        if not os.path.exists(self.history_file):
            atomic_write_json(self.history_file, {})

        # Pattern detection flag
        self.enable_pattern_detection = bool(enable_pattern_detection)

        # Modes
        self.mode_fast_scan = False
        self.mode_deep_scan = True  # default to deep scan for accuracy

        # Threadpool control
        self.max_workers = 6

    # ---- Watchlist / Discord helpers ----
    def set_watchlist(self, watchlist: List[str] | None) -> None:
        self.whitelist = list(watchlist or [])

    def set_discord_notifier(self, webhook_url: str | None, whitelist: List[str] | None = None) -> None:
        try:
            self.discord = DiscordNotifier(webhook_url)
            if whitelist:
                self.whitelist = list(whitelist)
        except Exception as e:
            logger.exception("Failed to init DiscordNotifier: %s", e)

    # ---- Caching helpers ----
    def _cache_get(self, key: str) -> Optional[dict]:
        with self._cache_lock:
            ts = self._cache_ts.get(key)
            if not ts:
                return None
            if time.time() - ts > self.cache_ttl:
                self._cache.pop(key, None)
                self._cache_ts.pop(key, None)
                return None
            return self._cache.get(key)

    def _cache_set(self, key: str, val: dict) -> None:
        self._cache[key] = val
        self._cache_ts[key] = time.time()

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_ts.clear()

    # ---- Robust data fetching with auto-repair (A) ----
    def get_stock_data(self, ticker: str, period: str = "3mo", interval: str = "1d") -> Optional[dict]:
        """
        Fetch using yfinance with caching and auto-repair strategies.
        Returns dict: {ticker, hist (DataFrame), info, financials, balance_sheet, cash_flow}
        """
        key = f"{ticker}|{period}|{interval}"
        cached = self._cache_get(key)
        if cached:
            return cached

        # Attempt fetch with retries & fallbacks
        attempts = 3
        last_exc = None
        for attempt in range(attempts):
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period=period, interval=interval, auto_adjust=False, prepost=False)
                if hist is None or hist.empty:
                    # if daily interval try 'max' & 1d fallback
                    if interval != "1d" and period != "max":
                        hist2 = tk.history(period="max", interval="1d", auto_adjust=False, prepost=False)
                        if hist2 is not None and not hist2.empty:
                            hist = hist2
                if hist is None or hist.empty:
                    raise ValueError("No historical data returned from yfinance")
                # Ensure index is datetime and sorted
                hist.index = pd.to_datetime(hist.index)
                hist = hist.sort_index()
                # Guard common missing columns
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in hist.columns:
                        hist[col] = np.nan
                # Try to fetch info safely
                info = {}
                financials = pd.DataFrame()
                balance = pd.DataFrame()
                cashflow = pd.DataFrame()
                try:
                    info = tk.info or {}
                    financials = tk.financials if hasattr(tk, "financials") else pd.DataFrame()
                    balance = tk.balance_sheet if hasattr(tk, "balance_sheet") else pd.DataFrame()
                    cashflow = tk.cash_flow if hasattr(tk, "cash_flow") else pd.DataFrame()
                except Exception:
                    pass

                result = {
                    "ticker": ticker,
                    "hist": hist,
                    "info": info,
                    "financials": financials,
                    "balance_sheet": balance,
                    "cash_flow": cashflow,
                }
                self._cache_set(key, result)
                return result
            except Exception as e:
                last_exc = e
                logger.debug("get_stock_data attempt %s failed for %s: %s", attempt + 1, ticker, e)
                time.sleep(0.5 + attempt * 0.5)
        logger.warning("get_stock_data final failure for %s: %s", ticker, last_exc)
        return None

    # ---- Technical indicators (base) ----
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        # Ensure numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                df[col] = np.nan
        # Fill small gaps
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

        # Simple moving averages
        df["MA_5"] = df["Close"].rolling(5, min_periods=1).mean()
        df["MA_20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["MA_50"] = df["Close"].rolling(50, min_periods=1).mean()

        # RSI 14 (Wilder-like via EWMA)
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)
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

        # Volume ratio
        df["Vol_MA_20"] = df["Volume"].rolling(20, min_periods=1).mean().replace(0, 1)
        df["Vol_Ratio"] = safe_div(df["Volume"], df["Vol_MA_20"])

        # ROC
        df["ROC_10"] = safe_div((df["Close"] - df["Close"].shift(10)), df["Close"].shift(10)) * 100

        # Support/resistance simple rolling
        df["Res_20"] = df["High"].rolling(20, min_periods=1).max()
        df["Sup_20"] = df["Low"].rolling(20, min_periods=1).min()
        return df

    # ---- Deep technical indicators ----
    def calculate_deep_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        for s in (8, 21, 50, 100, 200):
            df[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()

        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["TP"] = tp
        window_vwap = 20
        denom = df["Volume"].rolling(window=window_vwap, min_periods=1).sum().replace(0, 1e-9)
        df["VWAP"] = (tp * df["Volume"]).rolling(window=window_vwap, min_periods=1).sum() / denom

        # ATR 14
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR_14"] = (
            tr.rolling(14, min_periods=1)
            .mean()
            .replace(0, np.nan)
            .ffill()
            .fillna(1.0)
        )

        # OBV
        close_diff = df["Close"].diff().fillna(0)
        sign = np.sign(close_diff)
        df["OBV"] = (sign * df["Volume"]).cumsum().fillna(0)

        # ADX approximation (smoothed)
        up = df["High"].diff().fillna(0)
        down = -df["Low"].diff().fillna(0)
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        atr_s = tr.rolling(14, min_periods=1).mean().replace(0, 1e-9)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14, min_periods=1).sum() / atr_s)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14, min_periods=1).sum() / atr_s)
        dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        df["ADX_14"] = dx.rolling(14, min_periods=1).mean().fillna(0)

        df["EMA8_above_EMA21"] = (df["EMA_8"] > df["EMA_21"]).astype(int)
        df["EMA21_above_EMA50"] = (df["EMA_21"] > df["EMA_50"]).astype(int)
        return df

    # ---- Fundamental metrics (improved) ----
    def calculate_fundamental_metrics(self, stock_data: dict) -> dict:
        info = stock_data.get("info", {}) or {}
        fin = stock_data.get("financials", pd.DataFrame())
        current_price = info.get("currentPrice", info.get("regularMarketPrice", info.get("previousClose", float("nan"))))
        market_cap = info.get("marketCap", 0)
        trailing_eps = info.get("trailingEps", None) or info.get("eps", None) or 0
        pe = safe_div(float(current_price or 0), float(trailing_eps or 0)) if trailing_eps else (info.get("trailingPE") or 0)
        book_value = info.get("bookValue", None)
        pb = info.get("priceToBook", None) or (safe_div(float(current_price or 0), float(book_value)) if book_value else 0)
        roe = info.get("returnOnEquity", 0) or 0
        der = info.get("debtToEquity", 0) or 0
        div_yield = (info.get("dividendYield", 0) or 0) * 100

        # Sanitize & clamp ROE to -100..100 & convert to percent if needed
        try:
            if isinstance(roe, (int, float)):
                roe_val = float(roe)
                if abs(roe_val) < 5:
                    roe_pct = roe_val * 100
                else:
                    roe_pct = roe_val  # likely already percent
                roe_pct = max(min(roe_pct, 200.0), -200.0)
            else:
                roe_pct = 0.0
        except Exception:
            roe_pct = 0.0

        # Additional derived metrics safe
        return {
            "current_price": float(current_price) if pd.notna(current_price) else math.nan,
            "market_cap": market_cap,
            "pe_ratio": float(pe or 0),
            "pb_ratio": float(pb or 0),
            "eps": trailing_eps,
            "roe": float(roe_pct),
            "der": float(der or 0),
            "dividend_yield": float(div_yield),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
        }

    # ---- Macro context ----
    def analyze_macro_context(self) -> dict:
        symbols = {"IHSG": "^JKSE", "USDIDR": "USDIDR=X", "GOLD": "GC=F", "OIL": "CL=F"}
        macros = {}
        for name, sym in symbols.items():
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(period="5d", interval="1d")
                if hist is None or hist.empty:
                    macros[name] = {"symbol": sym, "error": "no_data"}
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
        if ihsg_pct > 0.35 and usd_pct < 0:
            sentiment = "Risk-on"
        elif ihsg_pct < -0.35 or usd_pct > 0.35:
            sentiment = "Risk-off"
        return {"macros": macros, "market_sentiment": sentiment}

    # ---- Score / recommender ----
    def score_stock(self, df: pd.DataFrame, fm: dict) -> dict:
        if df is None or df.empty:
            return {"score": 0.0, "components": {}, "signal": "NO_DATA"}

        latest = df.iloc[-1]
        # RSI
        rsi = float(latest.get("RSI_14", 50))
        rsi_score = max(0.0, 100.0 - abs(50.0 - rsi) * 2.0)

        # MACD hist
        macdh = float(latest.get("MACD_Hist", 0.0))
        macd_score = float(np.clip(50.0 + (macdh * 10.0), 0.0, 100.0))

        # MA trend
        ema8 = float(latest.get("EMA_8", 0.0))
        ema21 = float(latest.get("EMA_21", 0.0))
        ema50 = float(latest.get("EMA_50", 0.0))
        if ema8 > ema21 > ema50:
            ma_trend_score = 100.0
        elif ema8 < ema21 < ema50:
            ma_trend_score = 0.0
        else:
            ma_trend_score = 50.0

        # Volume surge
        vol_ratio = float(latest.get("Vol_Ratio", 1.0))
        vol_score = float(np.clip((vol_ratio - 1.0) * 50.0 + 50.0, 0.0, 100.0))

        # Composite
        total = (rsi_score * 0.2) + (macd_score * 0.3) + (ma_trend_score * 0.3) + (vol_score * 0.2)
        score = float(np.clip(total, 0.0, 100.0))

        if score >= 85:
            signal = "STRONG BUY"
        elif score >= 65:
            signal = "BUY"
        elif score >= 45:
            signal = "HOLD"
        else:
            signal = "SELL"

        components = {
            "rsi_score": rsi_score,
            "macd_score": macd_score,
            "ma_trend_score": ma_trend_score,
            "volume_score": vol_score,
        }
        return {"score": round(score, 2), "components": components, "signal": signal}

    # ---- 3-day plan improved (bidirectional) ----
    def generate_3day_decision(self, df: pd.DataFrame, fm: dict, capital: float = 100_000_000, risk_percent: float = 1.0) -> dict:
        if df is None or df.empty:
            return {"action": "NO_DATA"}
        latest = df.iloc[-1]
        entry = float(latest["Close"])
        atr = float(latest.get("ATR_14", 0.0) or (entry * 0.005))

        # support/resistance
        sup, res = self.detect_support_resistance(df, window=10, sensitivity=0.03)

        # For buy: stop below entry, for sell: stop above entry — we prepare both
        base_stop_buy = entry - 1.5 * atr
        base_stop_sell = entry + 1.5 * atr

        adaptive_stop_buy = max(base_stop_buy, sup * 0.98) if sup and sup < entry else base_stop_buy
        adaptive_stop_sell = min(base_stop_sell, res * 1.02) if res and res > entry else base_stop_sell

        R = 2.2
        base_target_buy = entry + R * (entry - adaptive_stop_buy)
        base_target_sell = entry - R * (adaptive_stop_sell - entry)

        adaptive_target_buy = min(base_target_buy, res * 0.995) if (res and res > entry) else base_target_buy
        adaptive_target_sell = max(base_target_sell, sup * 1.005) if (sup and sup < entry) else base_target_sell

        # momentum adjust
        if latest.get("EMA_8", 0) > latest.get("EMA_21", 0) > latest.get("EMA_50", 0):
            adaptive_target_buy *= 1.08
        elif latest.get("EMA_8", 0) < latest.get("EMA_21", 0) < latest.get("EMA_50", 0):
            adaptive_target_sell *= 0.92

        # choose action based on score + adx/roc
        score_pack = self.score_stock(df, fm)
        score = score_pack["score"]
        adx = float(latest.get("ADX_14", 0.0))
        roc = float(latest.get("ROC_10", 0.0))

        action = "HOLD"
        if score >= 80 and adx > 20:
            action = "ENTER_LONG"
        elif score >= 60 and roc > 0:
            action = "ENTER_LONG"
        elif score < 35 or adx < 12:
            action = "EXIT_OR_SHORT"

        # position sizing (for long scenario)
        risk_amount = capital * (risk_percent / 100.0)
        risk_per_share = max(1e-6, entry - adaptive_stop_buy)
        position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = position_size * entry

        return {
            "action": action,
            "entry": round(entry, 6),
            "stop": round(adaptive_stop_buy, 6),
            "target": round(adaptive_target_buy, 6),
            "entry_short": round(entry, 6),
            "stop_short": round(adaptive_stop_sell, 6),
            "target_short": round(adaptive_target_sell, 6),
            "position_size": position_size,
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "score": round(score, 2),
        }

    # ---- Support/resistance detection (robust) ----
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 10, sensitivity: float = 0.03) -> Tuple[Optional[float], Optional[float]]:
        if df is None or df.empty:
            return None, None
        dfc = df.copy()
        dfc["center_min"] = dfc["Low"].rolling(window=window, center=True, min_periods=1).min()
        dfc["center_max"] = dfc["High"].rolling(window=window, center=True, min_periods=1).max()
        unique_supports = sorted(dfc["center_min"].dropna().unique())
        unique_resistances = sorted(dfc["center_max"].dropna().unique())

        last_price = float(dfc["Close"].iloc[-1])
        nearby_supports = [s for s in unique_supports if s <= last_price * (1 + sensitivity)]
        nearby_resistances = [r for r in unique_resistances if r >= last_price * (1 - sensitivity)]

        nearest_support = float(max(nearby_supports)) if nearby_supports else None
        nearest_resistance = float(min(nearby_resistances)) if nearby_resistances else None
        return nearest_support, nearest_resistance

    # ---- Timing advice ----
    def timing_signal(self, last_price: float, nearest_support: float | None, nearest_resistance: float | None, main_signal: str) -> str:
        try:
            if nearest_support and last_price <= nearest_support * 1.01 and main_signal in ["BUY", "HOLD", "STRONG BUY", "ENTER_LONG"]:
                return f"🟢 Potensi Entry di sekitar support {nearest_support:.2f}"
            if nearest_resistance and last_price >= nearest_resistance * 0.99 and main_signal in ["SELL", "HOLD", "EXIT_OR_SHORT"]:
                return f"🔴 Potensi Take Profit di sekitar resistance {nearest_resistance:.2f}"
            if nearest_resistance and last_price > nearest_resistance:
                return f"🚀 Breakout di atas {nearest_resistance:.2f}"
            if nearest_support and last_price < nearest_support:
                return f"⚠️ Breakdown di bawah {nearest_support:.2f}"
        except Exception:
            pass
        return "⏸ Tidak ada sinyal timing signifikan saat ini."

    # ---- Intraday signals (multi-factor) ----
    def generate_intraday_signals(self, df: pd.DataFrame, lookback_ema_fast: int = 8, lookback_ema_slow: int = 21,
                                  macd_hist_threshold: float = 0.0, rsi_buy_thresh: float = 40, rsi_sell_thresh: float = 60,
                                  vol_spike_multiplier: float = 1.5) -> pd.DataFrame:
        # Gunakan vectorization
        df['ema_cross_up'] = (df['EMA_8'].shift(1) <= df['EMA_21'].shift(1)) & (df['EMA_8'] > df['EMA_21'])
        df['ema_cross_down'] = (df['EMA_8'].shift(1) >= df['EMA_21'].shift(1)) & (df['EMA_8'] < df['EMA_21'])
        
        if df is None or df.empty:
            return pd.DataFrame()
        dfc = df.copy().sort_index()
        if f"EMA_{lookback_ema_fast}" not in dfc.columns or f"EMA_{lookback_ema_slow}" not in dfc.columns:
            dfc = self.calculate_deep_technical_indicators(dfc)

        records = []
        prev = None
        for idx, row in dfc.iterrows():
            reasons = []
            confidence = 0.0
            ema_fast = row.get(f"EMA_{lookback_ema_fast}", np.nan)
            ema_slow = row.get(f"EMA_{lookback_ema_slow}", np.nan)
            if prev is not None:
                prev_fast = prev.get(f"EMA_{lookback_ema_fast}", np.nan)
                prev_slow = prev.get(f"EMA_{lookback_ema_slow}", np.nan)
                if prev_fast <= prev_slow and ema_fast > ema_slow:
                    reasons.append("EMA_CROSS_UP"); confidence += 0.25
                elif prev_fast >= prev_slow and ema_fast < ema_slow:
                    reasons.append("EMA_CROSS_DOWN"); confidence += 0.25
                prev_macd = prev.get("MACD_Hist", 0)
                curr_macd = row.get("MACD_Hist", 0)
                if prev_macd <= macd_hist_threshold and curr_macd > macd_hist_threshold:
                    reasons.append("MACD_TURN_POS"); confidence += 0.2
                elif prev_macd >= macd_hist_threshold and curr_macd < macd_hist_threshold:
                    reasons.append("MACD_TURN_NEG"); confidence += 0.2

                prev_close = prev.get("Close", np.nan)
                prev_vwap = prev.get("VWAP", np.nan)
                vwap = row.get("VWAP", np.nan)
                if not np.isnan(vwap) and not np.isnan(prev_vwap):
                    if prev_close <= prev_vwap and row.get("Close", np.nan) > vwap:
                        reasons.append("VWAP_CROSS_UP"); confidence += 0.15
                    elif prev_close >= prev_vwap and row.get("Close", np.nan) < vwap:
                        reasons.append("VWAP_CROSS_DOWN"); confidence += 0.15

            rsi = float(row.get("RSI_14", 50))
            if rsi < rsi_buy_thresh:
                reasons.append("RSI_OVERSOLD"); confidence += 0.05
            elif rsi > rsi_sell_thresh:
                reasons.append("RSI_OVERBOUGHT"); confidence += 0.05

            vol_ratio = float(row.get("Vol_Ratio", 1))
            if vol_ratio >= vol_spike_multiplier:
                reasons.append("VOL_SPIKE"); confidence += 0.1

            confidence = float(min(1.0, confidence))
            signal = "HOLD"
            if any(r in ("EMA_CROSS_UP", "MACD_TURN_POS", "VWAP_CROSS_UP") for r in reasons) and confidence >= 0.25:
                signal = "BUY"
            if any(r in ("EMA_CROSS_DOWN", "MACD_TURN_NEG", "VWAP_CROSS_DOWN") for r in reasons) and confidence >= 0.25:
                signal = "SELL"
            records.append({"timestamp": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                            "signal": signal, "reasons": ",".join(reasons), "confidence": confidence})
            prev = row
        try:
            df_signals = pd.DataFrame(records)
            df_signals["timestamp"] = pd.to_datetime(df_signals["timestamp"])
            df_signals = df_signals.set_index("timestamp")
            return df_signals
        except Exception:
            return pd.DataFrame(records)

    # ---- Simplified intraday backtest (toy) ----
    def backtest_intraday(self, df: pd.DataFrame, signals: pd.DataFrame, slippage: float = 0.0005, fee: float = 0.0005) -> dict:
        df.index = pd.to_datetime(df.index)
        signals.index = pd.to_datetime(signals.index)
        
        if df is None or df.empty or signals is None or signals.empty:
            return {"trades": [], "total_pnl": 0.0, "wins": 0, "losses": 0, "num_trades": 0}
        trades = []
        position = None
        for ts, s in signals.iterrows():
            # align to price index if exists
            if ts not in df.index:
                time_diffs = (df.index - ts).abs()
                nearest_idx = time_diffs.idxmin()
                price_row = df.loc[nearest_idx]
            else:
                price_row = df.loc[ts]
            price = float(price_row["Close"])
            if s["signal"] == "BUY" and position is None:
                entry_price = price * (1.0 + slippage + fee)
                position = {"entry_time": ts, "entry_price": entry_price, "confidence": float(s.get("confidence", 0.0)), "reasons": s.get("reasons", "")}
            elif s["signal"] == "SELL" and position is not None:
                exit_price = price * (1.0 - slippage - fee)
                pnl = exit_price - position["entry_price"]
                trades.append({**position, "exit_time": ts, "exit_price": exit_price, "pnl": pnl})
                position = None
        # close open
        if position is not None:
            last_price = float(df["Close"].iloc[-1])
            exit_price = last_price * (1.0 - slippage - fee)
            pnl = exit_price - position["entry_price"]
            trades.append({**position, "exit_time": df.index[-1], "exit_price": exit_price, "pnl": pnl})
        total_pnl = sum(t["pnl"] for t in trades) if trades else 0.0
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        return {"trades": trades, "total_pnl": total_pnl, "wins": wins, "losses": losses, "num_trades": len(trades)}


    def backtest_daily_tp(
        self,
        ticker: str,
        period: str = "3mo",
        interval: str = "60m",
        tp_pct: float = 0.0075,       # target 0.75% default (0.5-1% range)
        sl_pct: float = 0.004,        # stoploss 0.4% default
        horizon_candles: int = 48,    # max lookahead (e.g. 48 x 1h = 2 trading days)
        min_volume: int = 500_000,    # filter minimal likuiditas
        use_signals: bool = True      # gunakan generate_intraday_signals() sebagai entry
    ) -> dict:
        """
        Auto-backtest for daily TP/SL using intraday candles.
        Returns detailed trades and summary metrics.
        """
        stock_data = self.get_stock_data(ticker, period=period, interval=interval)
        if not stock_data:
            return {"error": "no_data", "ticker": ticker}

        df = stock_data.get("hist")
        if df is None or df.empty:
            return {"error": "no_data", "ticker": ticker}

        # prepare indicators
        df = df.sort_index().copy()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_deep_technical_indicators(df)

        # quick liquidity check on last available Vol_MA_20
        last_vol_ma = float(df.get("Vol_MA_20", pd.Series([0])).iloc[-1])
        if last_vol_ma < (min_volume or 0):
            return {"error": "low_liquidity", "ticker": ticker, "vol_ma": last_vol_ma}

        # generate signals (timestamp-indexed)
        if use_signals:
            signals = self.generate_intraday_signals(df)
        else:
            # fallback: treat every candle where EMA cross up occurs as BUY
            signals = pd.DataFrame([{"signal":"HOLD"}], index=[df.index[0]])

        trades = []
        # iterate signals chronologically
        for ts, s in signals.iterrows():
            sig = s.get("signal", "HOLD")
            if sig not in ("BUY", "SELL"):
                continue

            # align ts to df index (choose next available bar at or after ts)
            if ts not in df.index:
                later = df.index[df.index >= ts]
                if len(later) == 0:
                    continue
                entry_idx = df.index.get_loc(later[0])
            else:
                entry_idx = df.index.get_loc(ts)

            entry_price = float(df["Close"].iloc[entry_idx])
            # compute absolute price levels
            if sig == "BUY":
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

            # look forward up to horizon_candles
            end_idx = min(entry_idx + horizon_candles, len(df) - 1)
            forward_slice = df.iloc[entry_idx + 1: end_idx + 1]
            if forward_slice.empty:
                continue

            result = "LOSS"
            exit_price = float(df["Close"].iloc[end_idx])  # default exit at last available
            exit_time = df.index[end_idx]
            hit_first = None

            for idx, row in forward_slice.iterrows():
                # check order: whichever TP/SL touched first in chronological order
                if sig == "BUY":
                    if row["High"] >= tp_price:
                        result = "WIN"
                        exit_price = tp_price
                        exit_time = idx
                        hit_first = "TP"
                        break
                    if row["Low"] <= sl_price:
                        result = "LOSS"
                        exit_price = sl_price
                        exit_time = idx
                        hit_first = "SL"
                        break
                else:  # SELL
                    if row["Low"] <= tp_price:
                        result = "WIN"
                        exit_price = tp_price
                        exit_time = idx
                        hit_first = "TP"
                        break
                    if row["High"] >= sl_price:
                        result = "LOSS"
                        exit_price = sl_price
                        exit_time = idx
                        hit_first = "SL"
                        break

            pnl_pct = (exit_price - entry_price) / entry_price if sig == "BUY" else (entry_price - exit_price) / entry_price
            trades.append({
                "ticker": ticker,
                "signal": sig,
                "entry_time": df.index[entry_idx].isoformat(),
                "entry_price": round(entry_price, 6),
                "exit_time": exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time),
                "exit_price": round(exit_price, 6),
                "result": result,
                "hit_first": hit_first,
                "pnl_pct": round(pnl_pct * 100, 3),
                "holding_bars": int(df.index.get_loc(exit_time) - entry_idx) if exit_time in df.index else None
            })

        # summary stats
        total = len(trades)
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        winrate = round((wins / total * 100), 2) if total > 0 else 0.0
        avg_pnl = round((sum(t["pnl_pct"] for t in trades) / total), 3) if total > 0 else 0.0
        avg_win = round((sum(t["pnl_pct"] for t in trades if t["result"]=="WIN") / wins),3) if wins>0 else 0.0
        avg_loss = round((sum(t["pnl_pct"] for t in trades if t["result"]=="LOSS") / losses),3) if losses>0 else 0.0
        expectancy = round(((avg_win/100)* (wins/total) - (abs(avg_loss)/100)*(losses/total))*100,3) if total>0 else 0.0
        avg_holding = round((sum(t.get("holding_bars",0) for t in trades)/total),2) if total>0 else 0.0

        summary = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "winrate_pct": winrate,
            "avg_pnl_pct": avg_pnl,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "expectancy_pct": expectancy,
            "avg_holding_bars": avg_holding,
        }

        return {"summary": summary, "trades": trades}

    # ---- Demand analysis ----
    def analyze_demand(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {"demand_score": 0.0}

        last = df.iloc[-1]

        vol_ratio = float(last.get("Vol_Ratio", 1))
        atr = float(last.get("ATR_14", 0))
        close = float(last.get("Close", 0))
        high20 = float(df["High"].rolling(20).max().iloc[-1])

        # Demand components
        vol_demand = np.clip((vol_ratio - 1) * 40, 0, 40)
        atr_demand = np.clip((atr / close) * 300, 0, 30) if close > 0 else 0
        breakout_pressure = 30 if close >= high20 * 0.97 else 0

        demand_score = vol_demand + atr_demand + breakout_pressure

        return {
            "demand_score": round(demand_score, 2),
            "vol_ratio": round(vol_ratio, 2),
            "breakout_pressure": breakout_pressure > 0
        }

        # ---- Project activity estimation ----
    def analyze_project_activity(self, df: pd.DataFrame, fm: dict) -> dict:
        if df is None or df.empty:
            return {"project_score": 0, "active_projects": 0}

        last = df.iloc[-1]

        score = 0
        projects = 0

        # EMA trend = execution phase
        if last.get("EMA_8", 0) > last.get("EMA_21", 0) > last.get("EMA_50", 0):
            score += 25
            projects += 1

        # OBV accumulation
        if df["OBV"].iloc[-1] > df["OBV"].iloc[-10]:
            score += 20
            projects += 1

        # Volume sustainability
        if last.get("Vol_Ratio", 1) > 1.2:
            score += 20
            projects += 1

        # ATR stability = ongoing execution
        atr_mean = df["ATR_14"].rolling(20).mean().iloc[-1]
        if last.get("ATR_14", 0) >= atr_mean * 0.9:
            score += 15
            projects += 1

        # Market cap capacity
        mcap = fm.get("market_cap", 0)
        if mcap and mcap > 5_000_000_000_000:  # >5T IDR
            score += 20
            projects += 1

        return {
            "project_score": round(score, 2),
            "active_projects": projects
        }
    
    # ---- Single-stock analyze pipeline ----
    def analyze_one(self, ticker: str, period: str = "3mo", interval: str = "1d", capital: float = 100_000_000, risk_percent: float = 1.0) -> Optional[Dict[str, Any]]:
        stock_data = self.get_stock_data(ticker, period=period, interval=interval)
        if not stock_data:
            return None
        df = stock_data.get("hist")
        if df is None or df.empty:
            return None
        # ensure indices and numeric
        df = df.sort_index()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_deep_technical_indicators(df)
        fm = self.calculate_fundamental_metrics(stock_data)
        score_pack = self.score_stock(df, fm)

        demand = self.analyze_demand(df)
        project = self.analyze_project_activity(df, fm)

        plan = self.generate_3day_decision(df, fm, capital=capital, risk_percent=risk_percent)
        sup, res = self.detect_support_resistance(df)
        timing = self.timing_signal(float(df["Close"].iloc[-1]), sup, res, score_pack["signal"])
        intraday_signals = self.generate_intraday_signals(df)
        intraday_backtest = self.backtest_intraday(df, intraday_signals)
        patterns = self.detect_patterns(df) if self.enable_pattern_detection else {}

        out = {
            "ticker": ticker,
            "price_data": df,
            "fundamental_metrics": fm,
            "technical_score": score_pack,
            "3day_plan": plan,
            "support": sup,
            "resistance": res,
            "timing_advice": timing,
            "intraday_signals": intraday_signals,
            "intraday_backtest": intraday_backtest,
            "patterns": patterns,
            "demand_analysis": demand,
            "project_activity": project,
        }

        # Discord logic (cooldown & anti-spam)
        try:
            main_signal = score_pack.get("signal", "")
            action = plan.get("action", "")
            price = float(df["Close"].iloc[-1])
            alert_signal = None
            if main_signal == "STRONG BUY":
                alert_signal = "STRONG_BUY"
            elif action == "ENTER_LONG":
                alert_signal = "BUY"
            elif action == "EXIT_OR_SHORT":
                alert_signal = "SELL"
            # send only if ticker in whitelist and signal changed + cooldown
            if alert_signal and ticker in self.whitelist:
                with self._notify_lock:
                    prev = self._last_signal.get(ticker)
                    now = time.time()
                    last_sent = self._discord_cooldown.get(ticker, 0)
                    if prev != alert_signal and (now - last_sent) > self.discord_cooldown_seconds:
                        self._last_signal[ticker] = alert_signal
                        self._discord_cooldown[ticker] = now
                        # Compose embed
                        title = f"{'🚀' if 'BUY' in alert_signal else '🔴'} {alert_signal} — {ticker}"
                        desc = f"Price: {price:.2f}\nScore: {score_pack.get('score')}\nPlan: {plan.get('entry')} -> {plan.get('target')} / stop {plan.get('stop')}\n{timing}"
                        try:
                            self.discord.send_embed(title, desc, 0x16A34A if 'BUY' in alert_signal else 0xDC2626, ticker=ticker, signal=alert_signal)
                            self.discord.send_message(f"{alert_signal} {ticker} @ {price:.2f}", ticker=ticker, signal=alert_signal)
                            logger.info("Discord alert sent: %s %s", alert_signal, ticker)
                        except Exception as e:
                            logger.exception("Discord send failed: %s", e)
        except Exception as e:
            logger.exception("Discord logic failed: %s", e)

        return out

    # ---- Thread-safe history writer + evaluator (G) ----
    def evaluate_winloss(self, df, signal_index, entry, signal, target, stop, horizon=15):
        if signal_index not in df.index:
            return "UNKNOWN"
        
        idx = df.index.get_loc(signal_index)
        
        # Periksa batas
        if idx + horizon >= len(df):
            return "INSUFFICIENT_DATA"  # atau handle berbeda
        
        future = df.iloc[idx + 1 : min(idx + 1 + horizon, len(df))]
        if future.empty:
            return "UNKNOWN"

        for _, row in future.iterrows():
            high = row["High"]
            low = row["Low"]

            if signal == "BUY":
                if high >= target:
                    return "WIN"
                if low <= stop:
                    return "LOSS"

            elif signal == "SELL":
                if low <= target:
                    return "WIN"
                if high >= stop:
                    return "LOSS"

        return "LOSS"


    def _read_history(self) -> dict:
        with self._history_lock:
            return read_json_safe(self.history_file)

    def save_signal_history(self, ticker: str, signal: str, entry: float, target: float, stop: float, result: str) -> None:
        """
        Thread-safe append to history JSON. Also maintains a derived summary file for dashboard.
        """
        try:
            with self._history_lock:
                data = read_json_safe(self.history_file)
                if not isinstance(data, dict):
                    data = {}
                arr = data.get(ticker, [])
                arr.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal": signal,
                    "entry": float(entry),
                    "target": float(target),
                    "stop": float(stop),
                    "result": result
                })
                data[ticker] = arr
                atomic_write_json(self.history_file, data)
                # update summary
                try:
                    self._update_history_summary(data)
                except Exception:
                    logger.debug("Failed to update history summary")
        except Exception as e:
            logger.exception("save_signal_history failed: %s", e)

    def _update_history_summary(self, data: dict) -> None:
        """
        Build summary JSON including global summary, per_ticker stats, last N signals, accuracy by type.
        """
        summary = {"generated_at": datetime.now().isoformat(), "global_summary": {}, "per_ticker": {}, "last_signals": [], "accuracy_by_signal_type": {}}
        total = 0
        wins = 0
        losses = 0
        last_signals = []
        for t, records in data.items():
            wins_t = sum(1 for r in records if r.get("result") == "WIN")
            losses_t = sum(1 for r in records if r.get("result") == "LOSS")
            total_t = len(records)
            winrate_t = round((wins_t / total_t * 100), 2) if total_t > 0 else 0.0
            summary["per_ticker"][t] = {"wins": wins_t, "losses": losses_t, "total": total_t, "winrate": winrate_t}
            total += total_t
            wins += wins_t
            losses += losses_t
            last_signals.extend([{"ticker": t, **r} for r in records[-50:]])
        global_winrate = round((wins / total * 100), 2) if total > 0 else 0.0
        summary["global_summary"] = {"total_signals": total, "wins": wins, "losses": losses, "winrate": global_winrate}
        # accuracy by type (BUY/SELL)
        type_acc = {}
        for t, records in data.items():
            for r in records:
                tp = r.get("signal", "UNKNOWN")
                dt = type_acc.setdefault(tp, {"wins": 0, "total": 0})
                if r.get("result") == "WIN":
                    dt["wins"] += 1
                dt["total"] += 1
        for tp, v in type_acc.items():
            type_acc[tp] = {"wins": v["wins"], "total": v["total"], "winrate": round((v["wins"] / v["total"] * 100), 2) if v["total"] > 0 else 0.0}
        summary["accuracy_by_signal_type"] = type_acc
        # last signals chronologically
        last_signals_sorted = sorted(last_signals, key=lambda x: x.get("date", ""), reverse=True)[:200]
        summary["last_signals"] = last_signals_sorted
        # write summary file
        try:
            atomic_write_json(os.path.join(self.history_folder, "signal_history_summary.json"), summary)
        except Exception:
            logger.exception("Failed to write history summary")

    # ---- Pattern detection (H) basic implementations ----
    def detect_patterns(self, df: pd.DataFrame) -> dict:
        patterns = {}
        try:
            patterns["double_bottom"] = self._detect_double_bottom(df)
            patterns["double_top"] = self._detect_double_top(df)
            patterns["triangle"] = self._detect_triangle(df)
            # more patterns can be added
        except Exception:
            logger.debug("Pattern detection failed")
        return patterns

    def _detect_double_bottom(self, df: pd.DataFrame) -> bool:
        # simplistic: two local lows separated by a bounce
        if df is None or len(df) < 20:
            return False
        lows = df["Low"]
        idx = lows.argmin()
        # skip edges
        if idx < 3 or idx > len(df) - 4:
            return False
        left_min = lows.iloc[:idx].rolling(5).min().min()
        right_min = lows.iloc[idx+1:].rolling(5).min().min() if idx+1 < len(df) else np.inf
        center = lows.iloc[idx]
        return (abs(center - left_min) / max(1e-6, center) < 0.05) and (abs(center - right_min) / max(1e-6, center) < 0.05)

    def _detect_double_top(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < 20:
            return False
        highs = df["High"]
        idx = highs.argmax()
        if idx < 3 or idx > len(df) - 4:
            return False
        left_max = highs.iloc[:idx].rolling(5).max().max()
        right_max = highs.iloc[idx+1:].rolling(5).max().max() if idx+1 < len(df) else -np.inf
        center = highs.iloc[idx]
        return (abs(center - left_max) / max(1e-6, center) < 0.05) and (abs(center - right_max) / max(1e-6, center) < 0.05)

    def _detect_triangle(self, df: pd.DataFrame) -> bool:
        # simple slope convergence check on highs/lows over last 30 bars
        if df is None or len(df) < 30:
            return False
        recent = df[-30:]
        highs = recent["High"].values
        lows = recent["Low"].values
        # linear fit slopes
        x = np.arange(len(highs))
        try:
            high_coef = np.polyfit(x, highs, 1)[0]
            low_coef = np.polyfit(x, lows, 1)[0]
            # Opposing slopes with small magnitude suggest triangle
            return (abs(high_coef) < 0.5 and abs(low_coef) < 0.5 and np.sign(high_coef) != np.sign(low_coef))
        except Exception:
            return False

    # ---- Evaluate signals on historical data (winrate eval helper) ----
    def evaluate_signal_winrate(self, df: pd.DataFrame, signals: pd.DataFrame, horizon: int = 5, tp_pct: float = 0.01, sl_pct: float = 0.01) -> dict:
        """
        Evaluate signals dataframe (timestamp index) across a horizon of next N candles.
        Returns wins/losses/total/winrate.
        """
        if df is None or df.empty or signals is None or signals.empty:
            return {"total_signals": 0, "wins": 0, "losses": 0, "winrate": 0.0}
        dfc = df.copy().sort_index()
        wins = losses = total = 0
        for ts, s in signals.iterrows():
            sig = s.get("signal", "HOLD")
            if sig not in ("BUY", "SELL"):
                continue
            # find index
            if ts not in dfc.index:
                later = dfc.index[dfc.index > ts]
                if len(later) == 0:
                    continue
                entry_idx = dfc.index.get_loc(later[0])
            else:
                entry_idx = dfc.index.get_loc(ts)
            entry_price = float(dfc["Close"].iloc[entry_idx])
            end_idx = min(entry_idx + horizon, len(dfc) - 1)
            slice_df = dfc.iloc[entry_idx+1 : end_idx+1]
            if slice_df.empty:
                continue
            total += 1
            if sig == "BUY":
                tp = entry_price * (1 + tp_pct)
                sl = entry_price * (1 - sl_pct)
                hit_tp = (slice_df["High"] >= tp).any()
                hit_sl = (slice_df["Low"] <= sl).any()
                if hit_tp and not hit_sl:
                    wins += 1
                elif hit_sl and not hit_tp:
                    losses += 1
                else:
                    # which occurred first?
                    hit = "NONE"
                    for _, row in slice_df.iterrows():
                        if row["High"] >= tp:
                            hit = "TP"
                            break
                        if row["Low"] <= sl:
                            hit = "SL"
                            break
                    if hit == "TP":
                        wins += 1
                    elif hit == "SL":
                        losses += 1
                    else:
                        losses += 1
            else:
                tp = entry_price * (1 - tp_pct)
                sl = entry_price * (1 + sl_pct)
                hit_tp = (slice_df["Low"] <= tp).any()
                hit_sl = (slice_df["High"] >= sl).any()
                if hit_tp and not hit_sl:
                    wins += 1
                elif hit_sl and not hit_tp:
                    losses += 1
                else:
                    hit = "NONE"
                    for _, row in slice_df.iterrows():
                        if row["Low"] <= tp:
                            hit = "TP"
                            break
                        if row["High"] >= sl:
                            hit = "SL"
                            break
                    if hit == "TP":
                        wins += 1
                    elif hit == "SL":
                        losses += 1
                    else:
                        losses += 1
        winrate = round((wins / total * 100), 2) if total > 0 else 0.0
        return {"total_signals": total, "wins": wins, "losses": losses, "winrate": winrate}

    # ---- Batch runner (generate_recommendations) with modes and safe history tracking ----  period="3mo", interval="1h"
    def generate_recommendations(
        self, tickers:list, period: str, interval: str,
        top_n: int, capital=50_000_000, risk_percent=1.0, mode="deep"):

        results = {}
        ranked = []

        for ticker in tickers:
            try:
                data = self.analyze_one(ticker, period=period, interval=interval,
                                        capital=capital, risk_percent=risk_percent)

                if not data or data["price_data"].empty: continue

                df = data["price_data"]
                last = df.iloc[-1]

                # ========= FILTER WAJIB AGAR LAYAK TRADING ==========
                if last["Vol_MA_20"] < 500_000:                # volume minimal
                    continue
                if last["ATR_14"] < last["Close"]*0.005:       # volatilitas rendah
                    continue

                # ========= ENTRY BUY HIGH CONFIDENCE ==========
                cond_buy = (
                    last["EMA_8"] > last["EMA_21"] > last["EMA_50"] and
                    last["MACD_Hist"] > 0 and
                    last["RSI_14"] > 38 and last["RSI_14"] < 65 and
                    last["Vol_Ratio"] >= 1.3
                )

                # ========= ENTRY SELL ==========
                cond_sell = (
                    last["EMA_8"] < last["EMA_21"] < last["EMA_50"] and
                    last["MACD_Hist"] < 0 and
                    last["RSI_14"] > 60
                )

                # ========= TARGET DAY TRADE 0.5–1% ==========
                entry = last["Close"]
                tp = entry * 1.0075   # default +0.75%
                sl = entry * 0.996    # -0.4% risk
                rr = (tp-entry) / (entry-sl)
                score_boost = 1

                # reward saham volatil = prioritas
                if last["ATR_14"] > entry*0.009: score_boost += 0.3
                if last["Vol_Ratio"] > 1.8: score_boost += 0.2

                signal = "HOLD"
                if cond_buy: signal = "BUY"
                if cond_sell: signal = "SELL"

                # total_score = data["technical_score"]["score"]*0.7 + (score_boost*30)
                project_score = data["project_activity"]["project_score"]
                demand_score = data["demand_analysis"]["demand_score"]

                total_score = (
                    data["technical_score"]["score"] * 0.5 +
                    project_score * 0.3 +
                    demand_score * 0.2
                )

                results[ticker] = {
                    **data,
                    "day_trade":{
                        "signal": signal,
                        "entry": round(entry,3),
                        "tp": round(tp,3),
                        "sl": round(sl,3),
                        "RR": round(rr,2)
                    },
                    "rank_score": round(total_score,2)
                }

                if signal == "BUY":
                    ranked.append({
                        "ticker": ticker,
                        "score": total_score,
                        "projects": data["project_activity"]["active_projects"],
                        "signal": signal,
                        "entry": round(entry, 3),
                        "tp": round(tp, 3),
                        "sl": round(sl, 3)
                    })

            except Exception as e:
                pass

        ranked_sorted = sorted(
            ranked,
            key=lambda x: x["score"],
            reverse=True
        )[:top_n]

        return {
            "top": ranked_sorted,
            "results": {item["ticker"]: results[item["ticker"]] for item in ranked_sorted},
            "macro": self.analyze_macro_context()
        }


    # ---- Long-term simulation (unchanged mostly) ----
    def historical_cagr(self, df: pd.DataFrame, years: float) -> float:
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
        if df is None or df.empty:
            return 0.0
        rets = df["Close"].pct_change().dropna()
        if rets.empty:
            return 0.0
        daily_std = float(rets.std())
        return float(daily_std * math.sqrt(252))

    def simulate_long_term_investment(
        self,
        ticker: str,
        years: float,
        initial_capital: float,
        periodic_contribution: float,
        contribution_freq_per_year: int,
        n_simulations: int,
        method: str = "mc",
        analyst_estimates: Optional[list] = None,
        use_bootstrap: bool = True,
        seed: Optional[int] = None
    ) -> dict:

        # --- Fetch historical data ---
        try:
            stock_data = self.get_stock_data(ticker, period="max", interval="1d") or {}
            df = stock_data.get("hist")
        except Exception:
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
        annualized_mu = (1 + mu_daily)**252 - 1
        annualized_vol = sigma_daily * math.sqrt(252)
        hist_cagr = self.historical_cagr(df, years)

        rng = np.random.RandomState(seed or int(time.time()))
        steps_per_year = contribution_freq_per_year
        total_periods = int(round(years * steps_per_year))
        dt = 1.0 / steps_per_year

        # --- Analyst weighting ---
        analyst_results = []
        weighted_mu = annualized_mu
        weighted_vol = annualized_vol

        if analyst_estimates:
            total_w = 0.0
            agg_mu = 0.0
            for a in analyst_estimates:
                if isinstance(a, dict):
                    r = a.get("r", 0.07)
                    w = a.get("w", 1.0)
                else:
                    r = float(a)
                    w = 1.0

                total_w += w
                agg_mu += r * w

                final = initial_capital * ((1 + r)**years)

                if periodic_contribution:
                    m = contribution_freq_per_year
                    rate = r / m
                    if rate != 0:
                        fv_contrib = periodic_contribution * (((1 + rate)**(m * years) - 1) / rate)
                    else:
                        fv_contrib = periodic_contribution * m * years
                    final += fv_contrib

                analyst_results.append({
                    "annual_return": r,
                    "weight": w,
                    "final_value": final
                })

            weighted_mu = agg_mu / max(1e-9, total_w)

        # --- Deterministic mode ---
        if method == "deterministic":
            hist_proj = initial_capital * ((1 + hist_cagr)**years)

            if periodic_contribution:
                m = contribution_freq_per_year
                rate = hist_cagr / m if hist_cagr != 0 else 0

                if rate != 0:
                    fv_contrib = periodic_contribution * (((1 + rate)**(m * years) - 1) / rate)
                else:
                    fv_contrib = periodic_contribution * m * years

                hist_proj += fv_contrib

            return {
                "method": "deterministic",
                "projection": hist_proj,
                "historical_cagr": hist_cagr,
                "analyst_scenarios": analyst_results
            }

        # --- Monte Carlo Simulation ---
        simulations_final = []
        gbm_paths = [] if n_simulations <= 500 else None

        for _ in range(n_simulations):
            val = initial_capital
            path = [val]

            for _ in range(total_periods):

                # hybrid bootstrap + GBM
                if use_bootstrap and random.random() < 0.5:
                    r = rng.choice(
                        daily_rets.values,
                        size=max(1, int(252 / steps_per_year))
                    ).mean() * steps_per_year
                else:
                    drift = (weighted_mu - 0.5 * weighted_vol**2) * dt
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
        exp_cagr = (mean / initial_capital)**(1 / years) - 1
        sharpe = safe_div(annualized_mu - self.risk_free_rate, annualized_vol)

        sensitivity = {
            "+20% Return": mean * 1.2,
            "-20% Return": mean * 0.8,
            "+10% Vol": std * 1.1,
            "-10% Vol": std * 0.9
        }

        return {
            "method": "mc",
            "mc_percentiles": percentiles,
            "mc_mean": mean,
            "mc_std": std,
            "prob_loss": prob_loss,
            "expected_cagr": exp_cagr,
            "sharpe_ratio": sharpe,
            "sensitivity": sensitivity,
            "analyst_scenarios": analyst_results,
            "sample_paths": gbm_paths
        }

    def predict_price_direction(self, ticker: str, forecast_days: int = 5) -> dict:
        # Get data
        stock_data = self.get_stock_data(ticker, period="3mo", interval="1d")
        if not stock_data or stock_data["hist"].empty:
            return {"error": "no_data", "ticker": ticker}
        
        df = stock_data["hist"].copy().sort_index()
        df = self.calculate_technical_indicators(df)
        df = self.calculate_deep_technical_indicators(df)
        
        if len(df) < 30:
            return {"error": "insufficient_data", "ticker": ticker}
        
        # Analyze recent trend
        recent = df.tail(20)
        closes = recent["Close"].values
        
        # Calculate moving averages
        ma5 = recent["MA_5"].iloc[-1]
        ma20 = recent["MA_20"].iloc[-1]
        ma50 = recent["MA_50"].iloc[-1]
        
        # RSI analysis
        rsi = recent["RSI_14"].iloc[-1]
        
        # MACD analysis
        macd = recent["MACD"].iloc[-1]
        macd_signal = recent["MACD_Signal"].iloc[-1]
        
        # Volume analysis
        vol_ratio = recent["Vol_Ratio"].iloc[-1]
        
        # Calculate probabilities based on technicals
        up_prob = 0.5  # baseline 50%
        
        # Trend factor (25%)
        if ma5 > ma20 > ma50:
            up_prob += 0.25
        elif ma5 < ma20 < ma50:
            up_prob -= 0.25
        
        # RSI factor (15%)
        if rsi < 40:  # oversold
            up_prob += 0.15
        elif rsi > 60:  # overbought
            up_prob -= 0.15
        
        # MACD factor (10%)
        if macd > macd_signal:  # bullish crossover
            up_prob += 0.10
        elif macd < macd_signal:  # bearish crossover
            up_prob -= 0.10
        
        # Volume factor (10%)
        if vol_ratio > 1.5:  # volume spike
            # Volume spike with price up = strong bullish
            if closes[-1] > closes[-2]:
                up_prob += 0.10
            else:
                up_prob -= 0.10
        
        # Clamp probability
        up_prob = max(0.1, min(0.9, up_prob))
        down_prob = 1.0 - up_prob
        
        # Predict number of up/down days
        expected_up_days = int(round(forecast_days * up_prob))
        expected_down_days = forecast_days - expected_up_days
        
        # Calculate expected percentage moves
        # Use historical volatility for estimation
        returns = df["Close"].pct_change().dropna()
        avg_up_return = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.005  # 0.5% default
        avg_down_return = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.003  # 0.3% default
        
        # Adjust based on current momentum
        momentum = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
        if momentum > 0:
            avg_up_return *= (1 + momentum * 2)
        else:
            avg_down_return *= (1 + abs(momentum) * 2)
        
        # Ensure reasonable values
        avg_up_return = min(avg_up_return, 0.05)  # max 5% daily
        avg_down_return = min(avg_down_return, 0.04)  # max 4% daily
        
        # Net direction
        if up_prob > 0.6:
            net_dir = "UP"
            confidence = up_prob
        elif down_prob > 0.6:
            net_dir = "DOWN"
            confidence = down_prob
        else:
            net_dir = "SIDEWAYS"
            confidence = max(up_prob, down_prob)
        
        # Daily predictions
        daily_preds = []
        current_price = closes[-1]
        
        for day in range(1, forecast_days + 1):
            # Simple simulation
            day_up_prob = up_prob + (0.1 if day <= expected_up_days else -0.1)
            day_up_prob = max(0.3, min(0.7, day_up_prob))
            
            daily_preds.append({
                "day": day,
                "predicted_up": day_up_prob > 0.5,
                "up_probability": day_up_prob,
                "expected_pct": avg_up_return if day_up_prob > 0.5 else -avg_down_return,
                "confidence": min(day_up_prob, 1 - day_up_prob) * 2
            })
        
        return {
            "ticker": ticker,
            "forecast_days": forecast_days,
            "up_days": expected_up_days,
            "down_days": expected_down_days,
            "up_probability": round(up_prob, 3),
            "down_probability": round(down_prob, 3),
            "expected_up_pct": round(avg_up_return * 100, 2),  # in percentage
            "expected_down_pct": round(avg_down_return * 100, 2),  # in percentage
            "net_direction": net_dir,
            "confidence": round(confidence, 3),
            "current_price": round(current_price, 2),
            "daily_predictions": daily_preds,
            "technical_factors": {
                "trend": "BULLISH" if ma5 > ma20 > ma50 else "BEARISH" if ma5 < ma20 < ma50 else "SIDEWAYS",
                "rsi": round(rsi, 2),
                "macd_signal": "BULLISH" if macd > macd_signal else "BEARISH",
                "volume_status": "HIGH" if vol_ratio > 1.5 else "NORMAL"
            }
        }
#------ Robust 3-day prediction model (I) ------
def predict_3days(self, ticker: str) -> dict:
    """
    Hybrid model prediction:
    ARIMA + RandomForest + ATR + Macro sentiment
    Produces probability-weighted prediction for 3 future days.
    """

    # 1) Fetch data
    stock_data = self.get_stock_data(ticker, period="6mo", interval="1d")
    if not stock_data or stock_data["hist"].empty:
        return {"error": "no_data"}

    df = stock_data["hist"].copy().sort_index()
    df = self.calculate_technical_indicators(df)
    df = self.calculate_deep_technical_indicators(df)

    closes = df["Close"].dropna()

    if len(closes) < 50:
        return {"error": "insufficient_data"}

    # -----------------------
    # 2) ARIMA Forecast 3 Days
    # -----------------------
    try:
        model = ARIMA(closes.values, order=(3, 1, 2))
        model_fit = model.fit()
        arima_forecast = model_fit.forecast(steps=3)
    except Exception:
        # fallback
        arima_forecast = np.array([closes.iloc[-1]] * 3)

    # -----------------------
    # 3) ML: Random Forest Forecast
    # -----------------------
    df_ml = df.tail(120).copy()

    df_ml["return_1"] = df_ml["Close"].pct_change().shift(-1)
    df_ml["return_2"] = df_ml["Close"].pct_change(2).shift(-2)
    df_ml["return_3"] = df_ml["Close"].pct_change(3).shift(-3)
    df_ml = df_ml.dropna()

    FEATURES = [
        "Close", "RSI_14", "MACD_Hist", "EMA_8", "EMA_21", "EMA_50",
        "ATR_14", "Vol_Ratio", "ROC_10"
    ]

    if df_ml.shape[0] < 40:
        ml_preds = np.array([0, 0, 0])
    else:
        X = df_ml[FEATURES]
        y = df_ml["return_3"]

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            random_state=42
        )
        model.fit(X, y)

        last_feat = df_ml[FEATURES].iloc[-1].values.reshape(1, -1)
        pred_ret_3 = model.predict(last_feat)[0]

        # Simple return distribution
        ml_preds = closes.iloc[-1] * (1 + np.array([
            pred_ret_3 * 0.4,
            pred_ret_3 * 0.7,
            pred_ret_3 * 1.0,
        ]))

    # -----------------------
    # 4) ATR Volatility Adjustment
    # -----------------------
    last_close = float(closes.iloc[-1])
    atr = float(df["ATR_14"].iloc[-1])
    volatility_index = min(1.0, atr / last_close)

    # Volatility reduces confidence weight
    vol_penalty = (1 - volatility_index)

    # -----------------------
    # 5) Macro Sentiment Weight
    # -----------------------
    macro = self.analyze_macro_context()
    sentiment = macro.get("market_sentiment", "Neutral")

    if sentiment == "Risk-on":
        macro_w = 1.10
    elif sentiment == "Risk-off":
        macro_w = 0.85
    else:
        macro_w = 1.0

    # -----------------------
    # 6) HYBRID AGGREGATION
    # -----------------------
    hybrid = (arima_forecast * 0.55 + ml_preds * 0.45) * macro_w

    # uncertainty from ATR
    ci_upper = hybrid + (atr * 1.2)
    ci_lower = hybrid - (atr * 1.2)

    # -----------------------
    # 7) FINAL SIGNAL
    # -----------------------
    expected_return = (hybrid[-1] - last_close) / last_close * 100

    if expected_return > 1.5:
        final_signal = "BUY"
    elif expected_return < -1.0:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    confidence = max(0.05, min(1.0, vol_penalty * (abs(expected_return) / 2)))

    return {
        "ticker": ticker,
        "predicted_prices": [float(v) for v in hybrid],
        "ci_upper": [float(v) for v in ci_upper],
        "ci_lower": [float(v) for v in ci_lower],
        "expected_return_pct": round(float(expected_return), 3),
        "confidence": round(float(confidence), 3),
        "volatility_index": round(float(volatility_index), 3),
        "macro_bias": sentiment,
        "final_signal": final_signal
    }
