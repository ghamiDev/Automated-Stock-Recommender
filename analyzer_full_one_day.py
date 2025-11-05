from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import time
import math
import streamlit as st
import yfinance as yf

# Import base class (assumes analyzer_full.py is in PYTHONPATH)
try:
    from analyzer_full import AutomatedStockAnalyzer
except Exception as e:
    # If import fails, create a stub so file at least loads for editing.
    class AutomatedStockAnalyzer:
        def __init__(self, *args, **kwargs):
            pass


class AutomatedStockAnalyzerDailyTP(AutomatedStockAnalyzer):
    """Extension class that implements the TP-harian optimized logic."""

    def calculate_daily_bias(self, df: pd.DataFrame) -> str:
        """Daily bias based on VWAP and EMA8 vs EMA21.
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'.
        """
        if df is None or df.empty:
            return "NEUTRAL"
        latest = df.iloc[-1]
        try:
            close = latest["Close"]
            vwap = latest.get("VWAP", np.nan)
            ema8 = latest.get("EMA_8", np.nan)
            ema21 = latest.get("EMA_21", np.nan)
        except Exception:
            return "NEUTRAL"

        if pd.notna(vwap) and (close > vwap) and (ema8 > ema21):
            return "BULLISH"
        if pd.notna(vwap) and (close < vwap) and (ema8 < ema21):
            return "BEARISH"
        return "NEUTRAL"

    def adaptive_target_profit(self, entry: float, atr: float, vwap: float, price: float) -> float:
        """Adaptive target based on ATR and VWAP deviation.
        vwap_dev = (price - vwap) / vwap
        """
        base_tp = entry + 1.5 * atr
        if vwap and vwap > 0:
            vwap_dev = (price - vwap) / vwap
        else:
            vwap_dev = 0.0
        if vwap_dev > 0.02:
            base_tp *= 0.9
        elif vwap_dev < -0.02:
            base_tp *= 1.2
        return float(base_tp)

    def calculate_entry_probability(self, df: pd.DataFrame) -> float:
        """Return probability [0..1] based on EMA, MACD, RSI band.
        Weighted: EMA 0.4, MACD 0.4, RSI 0.2
        """
        if df is None or df.empty:
            return 0.0
        latest = df.iloc[-1]
        score = 0.0
        try:
            if latest.get("EMA_8", 0) > latest.get("EMA_21", 0):
                score += 0.4
            if latest.get("MACD_Hist", 0) > 0:
                score += 0.4
            rsi = latest.get("RSI_14", 50)
            if 45 < rsi < 65:
                score += 0.2
        except Exception:
            pass
        return round(min(1.0, max(0.0, score)), 3)

    def is_active_today(self, df: pd.DataFrame) -> bool:
        """Active if volume today > 1.2 * avg(10)
        Also ignore tiny-volume instruments (avg < 1e3) to avoid illiquid tickers.
        """
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

    def dynamic_stop(self, df: pd.DataFrame, entry: float, atr: float, support: Optional[float]) -> float:
        """Dynamic stop using ATR and recent session range (last 5 bars)."""
        try:
            if df is None or df.empty:
                return entry * 0.98
            last5 = df.tail(5)
            session_vol = (last5["High"].max() - last5["Low"].min()) if not last5.empty else 0.0
            dynamic_stop = entry - max(atr, session_vol * 0.5)
            if support and support < entry:
                adaptive_stop = max(dynamic_stop, support * 0.98)
            else:
                adaptive_stop = max(dynamic_stop, entry * 0.97)
            return float(adaptive_stop)
        except Exception:
            return float(entry * 0.97)

    def update_strategy_stats(self, daily_backtest_summary: Dict[str, Any]):
        """Simple feedback to tune cache_ttl or parameters.
        Keep it lightweight: adjust cache_ttl based on winrate.
        """
        try:
            wins = daily_backtest_summary.get("wins", 0)
            num = daily_backtest_summary.get("num_trades", 0)
            if num <= 0:
                return
            winrate = wins / max(1, num)
            if winrate < 0.35:
                # refresh data faster when poor performance
                self.cache_ttl = max(30, self.cache_ttl * 0.85)
            elif winrate > 0.7:
                self.cache_ttl = min(3600, self.cache_ttl * 1.05)
        except Exception:
            pass

    def generate_daily_tp_analysis(
        self,
        tickers: List[str],
        period: str = "1mo",
        interval: str = "60m",
        capital: float = 100_000_000,
        risk_percent: float = 1.0,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Generate 1-day optimized TP recommendations.

        Returns dict: {
            "results": {ticker: analysis_dict},
            "ranked": [(ticker, score), ...],
            "top": top_list
        }
        Each analysis_dict contains a '1day_plan' key with fields: action, entry, stop, target, prob
        """
        results = {}
        scored = []

        for t in tickers:
            try:
                data = self.get_stock_data(t, period=period, interval=interval)
                if not data:
                    continue
                df = data["hist"].copy().sort_index()
                if df is None or df.empty:
                    continue

                df = self.calculate_technical_indicators(df)
                df = self.calculate_deep_technical_indicators(df)

                # Skip illiquid/unmoving
                if not self.is_active_today(df):
                    # still store minimal info
                    fm = self.calculate_fundamental_metrics(data)
                    results[t] = {
                        "ticker": t,
                        "price_data": df,
                        "fundamental_metrics": fm,
                        "technical_score": self.score_stock(df, fm),
                        "1day_plan": {"action": "INACTIVE", "entry": None, "stop": None, "target": None, "prob": 0.0},
                        "support_resistance": self.detect_support_resistance(df),
                    }
                    continue

                fm = self.calculate_fundamental_metrics(data)
                score_pack = self.score_stock(df, fm)

                latest = df.iloc[-1]
                entry = float(latest["Close"])
                atr = float(latest.get("ATR_14", max(1.0, entry * 0.005)))

                sup, res = self.detect_support_resistance(df, window=10, sensitivity=0.03)

                # dynamic stop
                stop = self.dynamic_stop(df, entry, atr, sup)

                # adaptive target using vwap dev
                vwap = float(latest.get("VWAP", entry)) if latest.get("VWAP", np.nan) is not None else entry
                target = self.adaptive_target_profit(entry, atr, vwap, entry)

                # probability
                prob = self.calculate_entry_probability(df)

                # daily bias
                bias = self.calculate_daily_bias(df)

                # Adjust: accept only signals that are aligned with bias
                action = "HOLD"
                if score_pack.get("score", 0) >= 75 and prob >= 0.6 and bias == "BULLISH":
                    action = "ENTER_LONG"
                elif score_pack.get("score", 0) >= 65 and prob >= 0.6 and bias in ("BULLISH", "NEUTRAL"):
                    action = "ENTER_LONG"
                elif score_pack.get("score", 0) < 40 or bias == "BEARISH":
                    action = "EXIT_OR_SHORT"

                # Position sizing (same logic)
                risk_amount = capital * (risk_percent / 100.0)
                risk_per_share = entry - stop if (entry - stop) > 0 else 1e-9
                position_size = int(risk_amount / risk_per_share)
                position_value = position_size * entry

                one_day_plan = {
                    "action": action,
                    "entry": round(entry, 6),
                    "stop": round(stop, 6),
                    "target": round(target, 6),
                    "position_size": int(position_size),
                    "position_value": round(position_value, 2),
                    "risk_amount": round(risk_amount, 2),
                    "prob": float(prob),
                    "bias": bias,
                    "score": round(score_pack.get("score", 0), 2),
                }

                analysis = {
                    "ticker": t,
                    "price_data": df,
                    "fundamental_metrics": fm,
                    "technical_score": score_pack,
                    "1day_plan": one_day_plan,
                    "support_resistance": (sup, res),
                }

                # Keep for ranking
                results[t] = analysis
                scored.append((t, one_day_plan["score"]))

            except Exception as e:
                # non-fatal per ticker
                try:
                    st.warning(f"Error processing {t}: {e}")
                except Exception:
                    pass
                continue

        ranked_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        top = ranked_sorted[:top_n]
        return {"results": results, "ranked": ranked_sorted, "top": top}


# ---------------------------
# Streamlit renderer helper
# ---------------------------

def render_daily_tp_page(analyzer_obj: AutomatedStockAnalyzerDailyTP, tickers: List[str], period: str, interval: str, top_n: int, capital: float, risk_percent: float):
    """Render the new 'Analisa TP Harian' page. Mirrors existing Dashboard detail layout
    but replaces 3-Day Plan with 1-Day-Plan results from generate_daily_tp_analysis.

    Call this from gui_full.py when current_app_page == 'Analisa TP Harian'.
    """
    st.title("📆 Analisa TP Harian (Optimized)")
    st.markdown("Menjalankan analisa TP harian menggunakan logika adaptif (bias harian, probabilitas, TP adaptif, stop dinamis).")

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
                if "EXIT" in val or "SHORT" in val:
                    return "background-color:#dc2626;color:white"
                if "INACTIVE" in val:
                    return "background-color:#64748b;color:white"
            return ""

        st.dataframe(df_table.style.applymap(color_action, subset=["Action"]).set_properties(**{"font-family": "monospace"}), width='stretch')

        # Details: mirror the existing Dashboard expanders but show 1-Day-Plan instead of 3-Day
        st.subheader("Details (click expand untuk lihat tiap saham)")
        for ticker, _ in ranked:
            analysis = res_pack["results"].get(ticker)
            if not analysis:
                continue
            try:
                company_name = None
                try:
                    info = yf.Ticker(ticker).info
                    company_name = info.get("longName", "Unknown Company")
                except Exception:
                    company_name = "Unknown Company"

                one_day = analysis.get("1day_plan", {})
                sig = analysis["technical_score"]["signal"]
                trend_score = analysis["technical_score"]["components"].get("ma_trend_score", 50)
                emoji_trend = "⬆️" if trend_score > 60 else "⬇️" if trend_score < 40 else "↔️"
                expander_label = f"{ticker} — {company_name} — Score: {one_day.get('score', 0)} — Action: {one_day.get('action')} {emoji_trend}"

                with st.expander(expander_label, expanded=False):
                    cols = st.columns([2, 2, 1])
                    df = analysis["price_data"].copy().tail(120)

                    with cols[0]:
                        st.markdown("**Candlestick (mini)**")
                        import plotly.graph_objects as go
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
                            "MACD_Hist": round(ta.get("MACD_Hist", 4), 4),
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
                        plan = analysis.get("1day_plan", {})
                        st.write(f"Action: **{plan.get('action','N/A')}**")
                        st.write(f"Entry: {plan.get('entry','-')}")
                        st.write(f"Stop: {plan.get('stop','-')}")
                        st.write(f"Target: {plan.get('target','-')}")
                        st.write(f"Prob: {plan.get('prob',0):.2f}")
                        st.write(f"Position size: {plan.get('position_size','-')} (Value: {plan.get('position_value','-')})")

                        st.markdown("---")
                        st.markdown("⚡ Support–Resistance Timing")
                        sup, res = analysis.get("support_resistance", (None, None))
                        st.write(f"Nearest Support: **{sup if sup else 'N/A'}**")
                        st.write(f"Nearest Resistance: **{res if res else 'N/A'}**")
                        timing_advice = self_timing = ""
                        # Small timing logic
                        last_price = analysis["price_data"]["Close"].iloc[-1]
                        if sup and last_price <= sup * 1.01 and plan.get('action') in ["ENTER_LONG", "HOLD"]:
                            timing_advice = f"🟢 Potensi Entry di sekitar support {sup}"
                        elif res and last_price >= res * 0.99 and plan.get('action') in ["EXIT_OR_SHORT", "HOLD"]:
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
                    summary = (
                        f"{ticker} shows a {one_day.get('action','-')} bias with score {one_day.get('score',0):.1f}/100. "
                        f"Prob {one_day.get('prob',0):.2f}. "
                        f"Support {sup if sup else '-'} , Resistance {res if res else '-'} . "
                        f"{timing_advice} "
                        f"Consider entry near {one_day.get('entry','-')} with stop {one_day.get('stop','-')} and target {one_day.get('target','-')}.")
                    st.info(summary)

            except Exception as e:
                st.warning(f"Error rendering {ticker}: {e}")

    # optional: expose results download
    st.markdown("---")
    if st.button("⬇️ Download daily TP results (CSV)"):
        rows = []
        for t, analysis in res_pack["results"].items():
            last = None
            try:
                last = analysis["price_data"].iloc[-1]
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


# End of file
