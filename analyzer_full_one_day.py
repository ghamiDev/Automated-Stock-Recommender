
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import time
import math
import streamlit as st
import yfinance as yf

try:
    from analyzer_full import AutomatedStockAnalyzer
except Exception:
    class AutomatedStockAnalyzer:
        def __init__(self, *args, **kwargs):
            pass


class AutomatedStockAnalyzerDailyTP(AutomatedStockAnalyzer):
    """Optimized TP Harian — dengan perbaikan entry, stoploss, dan target profit."""

    # ------------------------
    # === LOGIKA DASAR ===
    # ------------------------

    def calculate_daily_bias(self, df: pd.DataFrame) -> str:
        """Bias harian berdasar VWAP & EMA crossover."""
        if df is None or df.empty:
            return "NEUTRAL"
        latest = df.iloc[-1]
        close = latest.get("Close", np.nan)
        vwap = latest.get("VWAP", np.nan)
        ema8 = latest.get("EMA_8", np.nan)
        ema21 = latest.get("EMA_21", np.nan)
        if pd.notna(vwap) and (close > vwap) and (ema8 > ema21):
            return "BULLISH"
        elif pd.notna(vwap) and (close < vwap) and (ema8 < ema21):
            return "BEARISH"
        return "NEUTRAL"

    def calculate_entry_probability(self, df: pd.DataFrame) -> float:
        """Probabilitas entry (0-1)."""
        if df is None or df.empty:
            return 0.0
        latest = df.iloc[-1]
        score = 0.0
        if latest.get("EMA_8", 0) > latest.get("EMA_21", 0):
            score += 0.4
        if latest.get("MACD_Hist", 0) > 0:
            score += 0.4
        rsi = latest.get("RSI_14", 50)
        if 45 < rsi < 65:
            score += 0.2
        return round(min(1.0, max(0.0, score)), 3)

    def is_active_today(self, df: pd.DataFrame) -> bool:
        """Aktif jika volume hari ini > 1.2x avg10."""
        if df is None or df.empty:
            return False
        try:
            vol_today = df["Volume"].iloc[-1]
            avg_vol = df["Volume"].rolling(10).mean().iloc[-1]
            if np.isnan(avg_vol) or avg_vol <= 0:
                return False
            return (vol_today > avg_vol * 1.2) and (avg_vol > 1000)
        except Exception:
            return False

    def dynamic_stop(self, df: pd.DataFrame, entry: float, atr: float, support: Optional[float], direction: str) -> float:
        """Stop loss dinamis sesuai arah posisi."""
        last5 = df.tail(5)
        session_vol = (last5["High"].max() - last5["Low"].min()) if not last5.empty else atr
        if direction == "LONG":
            base_stop = entry - max(atr, session_vol * 0.5)
            if support and support < entry:
                base_stop = max(base_stop, support * 0.98)
            if base_stop >= entry:  # validasi jaga jarak
                base_stop = entry - 1.2 * atr
        else:  # SHORT
            base_stop = entry + max(atr, session_vol * 0.5)
            if support and support > entry:
                base_stop = min(base_stop, support * 1.02)
            if base_stop <= entry:
                base_stop = entry + 1.2 * atr
        return float(base_stop)

    def adaptive_target_profit(self, entry: float, atr: float, vwap: float, direction: str, res: Optional[float], sup: Optional[float]) -> float:
        """Target profit adaptif sesuai arah & level S/R."""
        base_tp = 1.5 * atr
        if direction == "LONG":
            target = entry + base_tp
            if res and res > entry:
                target = min(target + 0.5 * atr, res * 0.98)
            if target <= entry:
                target = entry + 1.5 * atr
        else:  # SHORT
            target = entry - base_tp
            if sup and sup < entry:
                target = max(target - 0.5 * atr, sup * 1.02)
            if target >= entry:
                target = entry - 1.5 * atr
        return float(target)

    # ------------------------
    # === ANALISIS UTAMA ===
    # ------------------------

    def generate_daily_tp_analysis(
        self,
        tickers: List[str],
        period: str = "1mo",
        interval: str = "60m",
        capital: float = 100_000_000,
        risk_percent: float = 1.0,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Analisis TP Harian dengan entry/stop/target realistis."""
        results = {}
        scored = []

        for t in tickers:
            try:
                data = self.get_stock_data(t, period=period, interval=interval)
                if not data:
                    continue
                df = data["hist"].copy().sort_index()
                if df.empty:
                    continue

                df = self.calculate_technical_indicators(df)
                df = self.calculate_deep_technical_indicators(df)

                fm = self.calculate_fundamental_metrics(data)
                score_pack = self.score_stock(df, fm)
                score_val = score_pack.get("score", 0)

                # skip saham tidak aktif
                if not self.is_active_today(df):
                    results[t] = {
                        "ticker": t,
                        "price_data": df,
                        "fundamental_metrics": fm,
                        "technical_score": score_pack,
                        "1day_plan": {"action": "INACTIVE", "entry": None, "stop": None, "target": None, "prob": 0.0},
                        "support_resistance": self.detect_support_resistance(df),
                    }
                    continue

                latest = df.iloc[-1]
                entry = float(latest["Close"])
                atr = float(latest.get("ATR_14", max(1.0, entry * 0.005)))
                vwap = float(latest.get("VWAP", entry)) if not np.isnan(latest.get("VWAP", np.nan)) else entry
                sup, res = self.detect_support_resistance(df, window=10, sensitivity=0.03)

                prob = self.calculate_entry_probability(df)
                bias = self.calculate_daily_bias(df)

                # --- Arah posisi ---
                if score_val >= 70 and prob >= 0.6 and bias == "BULLISH":
                    action = "ENTER_LONG"
                elif score_val < 40 or bias == "BEARISH":
                    action = "ENTER_SHORT"
                else:
                    action = "HOLD"

                # --- Hitung Stop & Target ---
                if action == "ENTER_LONG":
                    stop = self.dynamic_stop(df, entry, atr, sup, "LONG")
                    target = self.adaptive_target_profit(entry, atr, vwap, "LONG", res, sup)
                elif action == "ENTER_SHORT":
                    stop = self.dynamic_stop(df, entry, atr, res, "SHORT")
                    target = self.adaptive_target_profit(entry, atr, vwap, "SHORT", res, sup)
                else:
                    stop = None
                    target = None

                # --- Position sizing ---
                risk_amount = capital * (risk_percent / 100.0)
                risk_per_share = abs(entry - stop) if stop else entry * 0.02
                position_size = int(risk_amount / risk_per_share)
                position_value = position_size * entry

                one_day_plan = {
                    "action": action,
                    "entry": round(entry, 6),
                    "stop": round(stop, 6) if stop else None,
                    "target": round(target, 6) if target else None,
                    "position_size": position_size,
                    "position_value": round(position_value, 2),
                    "risk_amount": round(risk_amount, 2),
                    "prob": float(prob),
                    "bias": bias,
                    "score": round(score_val, 2),
                }

                results[t] = {
                    "ticker": t,
                    "price_data": df,
                    "fundamental_metrics": fm,
                    "technical_score": score_pack,
                    "1day_plan": one_day_plan,
                    "support_resistance": (sup, res),
                }
                scored.append((t, one_day_plan["score"]))

            except Exception as e:
                try:
                    st.warning(f"Error processing {t}: {e}")
                except Exception:
                    pass
                continue

        ranked_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        top = ranked_sorted[:top_n]
        return {"results": results, "ranked": ranked_sorted, "top": top}