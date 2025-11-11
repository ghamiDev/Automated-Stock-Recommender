# analyzer_daily_tp_fixed_full.py
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import time
import math
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Try import base analyzer (your project). If not available, provide a stub so file loads.
try:
    from analyzer_full import AutomatedStockAnalyzer
except Exception:
    class AutomatedStockAnalyzer:
        def __init__(self, *args, **kwargs):
            # minimal attributes some code expects
            self.cache_ttl = 300
            # optional discord notifier stub
            self.discord = type("D", (), {"send_watchlist_added": lambda self, t: None})()

        # minimal stubs for methods used
        def get_stock_data(self, ticker, period="1mo", interval="60m"):
            return None
        def calculate_technical_indicators(self, df): return df
        def calculate_deep_technical_indicators(self, df): return df
        def calculate_fundamental_metrics(self, data): return {}
        def score_stock(self, df, fm): return {"score": 50, "signal": "NEUTRAL", "components": {"ma_trend_score": 50}}
        def detect_support_resistance(self, df, window=10, sensitivity=0.03): return (None, None)


class AutomatedStockAnalyzerDailyTP(AutomatedStockAnalyzer):
    """Optimized TP Harian — perbaikan entry/stop/target (long & short supported)."""

    # ------------------------
    # Helper / indicator-based decisions
    # ------------------------
    def calculate_daily_bias(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "NEUTRAL"
        latest = df.iloc[-1]
        close = latest.get("Close", np.nan)
        vwap = latest.get("VWAP", np.nan)
        ema8 = latest.get("EMA_8", np.nan)
        ema21 = latest.get("EMA_21", np.nan)
        if pd.notna(vwap) and (close > vwap) and (ema8 > ema21):
            return "BULLISH"
        if pd.notna(vwap) and (close < vwap) and (ema8 < ema21):
            return "BEARISH"
        return "NEUTRAL"

    def calculate_entry_probability(self, df: pd.DataFrame) -> float:
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
        if df is None or df.empty:
            return False
        try:
            vol_today = float(df["Volume"].iloc[-1])
            avg_vol = float(df["Volume"].rolling(10).mean().iloc[-1])
            if np.isnan(avg_vol) or avg_vol <= 0:
                return False
            return (vol_today > avg_vol * 1.2) and (avg_vol > 1000)
        except Exception:
            return False

    # ------------------------
    # Stop & target calculaton (direction-aware)
    # ------------------------
    def dynamic_stop(self, df: pd.DataFrame, entry: float, atr: float, sr_level: Optional[float], direction: str) -> float:
        """Calculate dynamic stop depending on direction.
        For LONG: stop below entry. For SHORT: stop above entry.
        sr_level is nearest S or R depending on context (support for long, resistance for short).
        """
        try:
            last5 = df.tail(5)
            session_range = (last5["High"].max() - last5["Low"].min()) if not last5.empty else atr
        except Exception:
            session_range = atr

        if direction == "LONG":
            base_stop = entry - max(atr, session_range * 0.5)
            # if there's a support level use it as floor (slightly below)
            if sr_level and sr_level < entry:
                base_stop = max(base_stop, sr_level * 0.98)
            # safety: ensure stop < entry
            if base_stop >= entry:
                base_stop = entry - (1.2 * atr)
        else:  # SHORT
            base_stop = entry + max(atr, session_range * 0.5)
            # if there's a resistance use it
            if sr_level and sr_level > entry:
                base_stop = min(base_stop, sr_level * 1.02)
            # safety: ensure stop > entry
            if base_stop <= entry:
                base_stop = entry + (1.2 * atr)

        return float(base_stop)

    def adaptive_target_profit(self, entry: float, atr: float, vwap: float, direction: str, res: Optional[float], sup: Optional[float]) -> float:
        """Adaptive target that ensures for LONG: target > entry, for SHORT: target < entry.
        Uses ATR base and tries to respect nearest S/R to avoid impossible targets.
        """
        base_tp = 1.5 * atr
        if direction == "LONG":
            # Prefer resistance if exists
            if res and res > entry:
                # aim below resistance but above entry
                target = min(entry + base_tp + 0.5 * atr, res * 0.98)
            else:
                target = entry + base_tp
            # safety
            if target <= entry:
                target = entry + (1.5 * atr)
        else:
            # SHORT
            if sup and sup < entry:
                target = max(entry - base_tp - 0.5 * atr, sup * 1.02)
            else:
                target = entry - base_tp
            if target >= entry:
                target = entry - (1.5 * atr)
        return float(target)

    # ------------------------
    # Main analysis routine
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
        results: Dict[str, Any] = {}
        scored: List[Tuple[str, float]] = []

        for t in tickers:
            try:
                data = self.get_stock_data(t, period=period, interval=interval)
                if not data:
                    continue
                df = data.get("hist")
                if df is None or df.empty:
                    continue
                df = df.copy().sort_index()

                # indicators
                df = self.calculate_technical_indicators(df)
                df = self.calculate_deep_technical_indicators(df)

                fm = self.calculate_fundamental_metrics(data)
                score_pack = self.score_stock(df, fm)
                score_val = score_pack.get("score", 0)

                # skip illiquid/unmoving
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
                # entry is last close
                entry = float(latest.get("Close", 0.0))
                # ATR fallback to small percent of price to avoid zero
                atr = float(latest.get("ATR_14", max(1.0, entry * 0.005)))
                vwap = float(latest.get("VWAP", entry)) if not np.isnan(latest.get("VWAP", np.nan)) else entry
                sup, res = self.detect_support_resistance(df, window=10, sensitivity=0.03)

                prob = self.calculate_entry_probability(df)
                bias = self.calculate_daily_bias(df)

                # --- Decide action (LONG / SHORT / HOLD)
                # Rules: prefer long when score high & bias bullish, short when weak or bearish
                if score_val >= 70 and prob >= 0.6 and bias == "BULLISH":
                    action = "ENTER_LONG"
                elif score_val < 40 or bias == "BEARISH":
                    # allow short
                    action = "ENTER_SHORT"
                else:
                    action = "HOLD"

                # --- Calculate stop & target depending on action
                stop = None
                target = None
                if action == "ENTER_LONG":
                    stop = self.dynamic_stop(df, entry, atr, sup, "LONG")
                    target = self.adaptive_target_profit(entry, atr, vwap, "LONG", res, sup)
                elif action == "ENTER_SHORT":
                    # note: for dynamic_stop we pass nearest resistance as sr_level for short
                    stop = self.dynamic_stop(df, entry, atr, res, "SHORT")
                    target = self.adaptive_target_profit(entry, atr, vwap, "SHORT", res, sup)

                # Validate numeric and meaningful relationships
                if stop is not None:
                    # ensure stop and target are finite numbers
                    if not np.isfinite(stop):
                        stop = None
                if target is not None:
                    if not np.isfinite(target):
                        target = None

                # Extra safety: ensure LONG -> target > entry; SHORT -> target < entry
                if action == "ENTER_LONG" and target is not None and target <= entry:
                    target = entry + 1.5 * atr
                if action == "ENTER_SHORT" and target is not None and target >= entry:
                    target = entry - 1.5 * atr

                # Position sizing (risk management)
                risk_amount = float(capital) * (float(risk_percent) / 100.0)
                if stop is not None:
                    risk_per_share = abs(entry - stop)
                    if risk_per_share <= 0:
                        risk_per_share = max(1e-6, entry * 0.01)
                else:
                    # fallback if no stop calculated: assume 2% risk per share
                    risk_per_share = max(1e-6, entry * 0.02)

                position_size = int(max(0, math.floor(risk_amount / risk_per_share)))
                position_value = position_size * entry

                one_day_plan = {
                    "action": action,
                    "entry": round(entry, 6),
                    "stop": round(stop, 6) if stop is not None else None,
                    "target": round(target, 6) if target is not None else None,
                    "position_size": int(position_size),
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


# ---------------------------
# Streamlit renderer helper (integrated)
# ---------------------------

def render_daily_tp_page(analyzer_obj: AutomatedStockAnalyzerDailyTP, tickers: List[str], period: str, interval: str, top_n: int, capital: float, risk_percent: float):
    st.title("📆 Analisa TP Harian (Optimized & Fixed)")
    st.markdown("Analisa TP harian — perbaikan entry/stop/target (support long & short).")

    if st.button("▶️ Generate Daily TP Recommendations"):
        with st.spinner("Menganalisa tickers untuk TP harian..."):
            res_pack = analyzer_obj.generate_daily_tp_analysis(
                tickers=tickers,
                period=period,
                interval=interval,
                capital=capital,
                risk_percent=risk_percent,
                top_n=top_n,
            )
            st.session_state["daily_tp_results"] = res_pack
            st.session_state["daily_tp_run"] = time.time()
            st.rerun()

    res_pack = st.session_state.get("daily_tp_results", None)
    if res_pack is None:
        st.info("Belum ada hasil TP harian. Klik 'Generate Daily TP Recommendations' untuk memulai.")
        return

    ranked = res_pack.get("top", [])
    st.subheader("Top Daily TP Recommendations")

    if not ranked:
        st.warning("Tidak ada rekomendasi harian.")
    else:
        rows = []
        for i, (ticker, score) in enumerate(ranked, start=1):
            analysis = res_pack["results"].get(ticker)
            if not analysis:
                continue
            last_price = None
            try:
                last_price = analysis["price_data"]["Close"].iloc[-1]
            except Exception:
                last_price = None
            one_day = analysis.get("1day_plan", {})
            rows.append({
                "No": i,
                "Ticker": ticker,
                "Price": round(float(last_price), 2) if last_price is not None else None,
                "Prob": one_day.get("prob"),
                "Bias": one_day.get("bias"),
                "Action": one_day.get("action"),
                "Entry": one_day.get("entry"),
                "Stop": one_day.get("stop"),
                "Target": one_day.get("target"),
                "Score": one_day.get("score"),
            })

        df_table = pd.DataFrame(rows)

        def color_action(val):
            if isinstance(val, str):
                if "ENTER_LONG" in val:
                    return "background-color:#16a34a;color:white"
                if "ENTER_SHORT" in val or "EXIT" in val:
                    return "background-color:#dc2626;color:white"
                if "INACTIVE" in val:
                    return "background-color:#64748b;color:white"
            return ""

        st.dataframe(df_table.style.applymap(color_action, subset=["Action"]).set_properties(**{"font-family": "monospace"}), width='stretch')

        st.subheader("Details (klik expand untuk setiap saham)")
        for ticker, _ in ranked:
            analysis = res_pack["results"].get(ticker)
            if not analysis:
                continue
            try:
                # company lookup (best-effort)
                company_name = "Unknown Company"
                try:
                    info = yf.Ticker(ticker).info
                    company_name = info.get("longName", company_name)
                except Exception:
                    pass

                plan = analysis.get("1day_plan", {})
                sig = analysis["technical_score"].get("signal", "N/A")
                trend_score = analysis["technical_score"].get("components", {}).get("ma_trend_score", 50)
                emoji_trend = "⬆️" if trend_score > 60 else "⬇️" if trend_score < 40 else "↔️"
                expander_label = f"{ticker} — {company_name} — Score: {plan.get('score', 0)} — Action: {plan.get('action')} {emoji_trend}"

                with st.expander(expander_label, expanded=False):
                    cols = st.columns([2, 2, 1])
                    df = analysis["price_data"].copy().tail(120)

                    with cols[0]:
                        st.markdown("**Candlestick (mini)**")
                       
                        if not df.empty:
                            fig = go.Figure(data=[go.Candlestick(
                                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
                            )])
                            sup, res = analysis.get("support_resistance", (None, None))
                            if sup:
                                fig.add_hline(y=sup, line_dash="dot", line_color="green",
                                              annotation_text=f"Support {sup}", annotation_position="bottom right")
                            if res:
                                fig.add_hline(y=res, line_dash="dot", line_color="red",
                                              annotation_text=f"Resistance {res}", annotation_position="top right")
                            for ma in ["MA_5", "MA_20", "MA_50", "EMA_8", "EMA_21"]:
                                if ma in df.columns:
                                    fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1)))
                            fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
                            st.plotly_chart(fig, config={"displaylogo": False, "responsive": True}, width='stretch')
                        else:
                            st.write("No price data to show.")

                    with cols[1]:
                        st.markdown("**Indicators Summary**")
                        ta = analysis["price_data"].iloc[-1].to_dict()
                        fm = analysis["fundamental_metrics"]
                        items = {
                            "Close": ta.get("Close"),
                            "RSI_14": round(ta.get("RSI_14", 0), 2),
                            "MACD_Hist": round(ta.get("MACD_Hist", 0), 4),
                            "Vol_Ratio": round(ta.get("Vol_Ratio", 1), 2),
                            "ATR_14": round(ta.get("ATR_14", 0), 4),
                            "EMA8>EMA21": bool(ta.get("EMA8_above_EMA21", 0)),
                        }
                        st.dataframe(pd.DataFrame(list(items.items()), columns=["Metric", "Value"]), width="stretch")
                        st.markdown("**Fundamental (quick)**")
                        fm_quick = {"Price": fm.get("current_price"), "P/E": fm.get("pe_ratio"),
                                    "P/B": fm.get("pb_ratio"), "ROE": fm.get("roe"), "DER": fm.get("der")}
                        st.dataframe(pd.DataFrame(list(fm_quick.items()), columns=["Metric", "Value"]), width="stretch")

                    with cols[2]:
                        st.markdown("**1-Day Plan**")
                        st.write(f"Action: **{plan.get('action','N/A')}**")
                        st.write(f"Entry: {plan.get('entry','-')}")
                        st.write(f"Stop: {plan.get('stop','-')}")
                        st.write(f"Target: {plan.get('target','-')}")
                        st.write(f"Prob: {plan.get('prob',0):.2f}")
                        st.write(f"Position size: {plan.get('position_size','-')} (Value: {plan.get('position_value','-')})")

                        st.markdown("---")
                        st.markdown("⚡ Support–Resistance Timing")
                        sup, res = analysis.get("support_resistance", (None, None))
                        last_price = analysis["price_data"]["Close"].iloc[-1]
                        timing_advice = ""
                        if sup and last_price <= sup * 1.01 and plan.get('action') in ["ENTER_LONG", "HOLD"]:
                            timing_advice = f"🟢 Potensi Entry di sekitar support {sup}"
                        elif res and last_price >= res * 0.99 and plan.get('action') in ["ENTER_SHORT", "HOLD"]:
                            timing_advice = f"🔴 Potensi Take Profit di sekitar resistance {res}"
                        elif res and last_price > res:
                            timing_advice = f"🚀 Breakout di atas {res}"
                        elif sup and last_price < sup:
                            timing_advice = f"⚠️ Breakdown di bawah {sup}"
                        else:
                            timing_advice = "⏸ Tidak ada sinyal timing signifikan saat ini."
                        st.info(timing_advice)

                        st.markdown("---")
                        if st.button(f"➕ Add {ticker} to Watchlist", key=f"wl_daily_{ticker}"):
                            if ticker not in st.session_state.get("watchlist", []):
                                st.session_state.setdefault("watchlist", []).append(ticker)
                                try:
                                    analyzer_obj.discord.send_watchlist_added(ticker)
                                except Exception:
                                    pass
                                st.success(f"{ticker} added to watchlist")
                            else:
                                st.info(f"{ticker} is already in watchlist")

                    st.markdown("**AI Summary (templated)**")
                    one_day = analysis.get("1day_plan", {})
                    sup, res = analysis.get("support_resistance", (None, None))
                    summary = (
                        f"{ticker} shows a {one_day.get('action','-')} bias with score {one_day.get('score',0):.1f}/100. "
                        f"Prob {one_day.get('prob',0):.2f}. "
                        f"Support {sup if sup else '-'} , Resistance {res if res else '-'} . "
                        f"{timing_advice} "
                        f"Consider entry near {one_day.get('entry','-')} with stop {one_day.get('stop','-')} and target {one_day.get('target','-')}."
                    )
                    st.info(summary)

            except Exception as e:
                st.warning(f"Error rendering {ticker}: {e}")

    # download CSV
    st.markdown("---")
    if st.button("⬇️ Download daily TP results (CSV)"):
        rows = []
        for t, analysis in res_pack["results"].items():
            last = {}
            try:
                last = analysis["price_data"].iloc[-1].to_dict()
            except Exception:
                last = {}
            plan = analysis.get("1day_plan", {})
            rows.append({
                "ticker": t,
                "close": last.get("Close"),
                "prob": plan.get("prob"),
                "score": plan.get("score"),
                "action": plan.get("action"),
                "entry": plan.get("entry"),
                "stop": plan.get("stop"),
                "target": plan.get("target"),
            })
        out_df = pd.DataFrame(rows)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="daily_tp_recommendations.csv", mime="text/csv")