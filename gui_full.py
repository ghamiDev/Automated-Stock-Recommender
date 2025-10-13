import yfinance as yf
import streamlit as st
import pandas as pd
from analyzer_full import AutomatedStockAnalyzer
from datetime import datetime
import plotly.graph_objects as go
from idx_listed import get_idx_tickers

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Automated Stock Recommender", layout="wide", page_icon="📈")

# CSS dark theme minor adjustments
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f14; color: #e6eef6; }
    .css-1d391kg { background-color: #0b0f14; }
    .stSidebar { background-color: #0f1418; }
    .stButton>button { background-color: #1f2933; color: #e6eef6; }
    .stDownloadButton>button { background-color: #2b6cb0; color: #fff; }
    thead th { background-color: #111827 !important; color: #e6eef6 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# App header + controls
# ---------------------------
st.title("📈 Automated Stock Recommender")
st.markdown("Real-time-ish recommendation engine (yfinance). Dashboard shows top recommended stocks and expandable detail per stock.")


# Sidebar controls
with st.sidebar:
    st.header("Controls")

    # ambil otomatis dari IDX (cached 1 hari)
    try:
        tickers_list = get_idx_tickers()
        default_list = ", ".join(tickers_list)
    except:
        st.warning("⚠️ Gagal memuat daftar IDX otomatis, fallback ke default list.")
        default_list = "BBCA.JK, BBRI.JK, BMRI.JK, TLKM.JK, ASII.JK, UNVR.JK, ICBP.JK, INDF.JK, TPIA.JK, EXCL.JK,ADRO.JK, ANTM.JK, BRPT.JK, ITMG.JK, PGAS.JK,AKRA.JK, SMGR.JK, GGRM.JK, JSMR.JK, KLBF.JK,MNCN.JK, WIKA.JK, WSKT.JK, CPIN.JK, ULTJ.JK,BMTR.JK, BRIS.JK, BTPS.JK, INCO.JK, MDKA.JK,SIDO.JK, TINS.JK, TOWR.JK, BUKA.JK, EMTK.JK,ARTO.JK, BYAN.JK, HRUM.JK, JPFA.JK, MYOR.JK"

    # tickers_input = st.text_area("Tickers (comma separated)", value=default_list, height=140)
    tickers = [t.strip().upper() for t in default_list.split(",") if t.strip()]
    period_choice = st.selectbox("Period for analysis", ["1mo", "3mo", "6mo", "1y"], index=0)
    interval_choice = st.selectbox("Interval (yfinance)", ["60m", "90m", "1d"], index=0)
    top_n = st.number_input("Top N recommendations", min_value=1, max_value=50, value=5, step=1)
    # capital = st.number_input("Capital (currency)", value=100000000, step=1000000)
    risk_percent = st.slider("Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    auto_refresh = st.checkbox("Auto refresh (simple)", value=False)
    refresh_interval = st.number_input("Auto refresh interval (seconds)", min_value=30, max_value=3600, value=600, step=10)
    st.markdown("---")
    if st.button("🔄 Refresh now"):
        st.rerun()

# session state for watchlist
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []

# Mode pilihan
st.markdown("### ⚙️ Pilih Mode Analisis")
analysis_mode = st.radio(
    "Mode analisis:",
    ["All Analysis (Full Auto)", "Single Analysis (Manual)"],
    index=0,
    horizontal=True
)

# --- Jika mode SINGLE, tampilkan dropdown saham ---
selected_tickers = []
if analysis_mode == "Single Analysis (Manual)":
    try:
        from idx_listed import get_idx_tickers
        idx_tickers = get_idx_tickers()
    except Exception:
        idx_tickers = tickers  # fallback

    st.markdown("### 🧩 Pilih Satu atau Beberapa Saham untuk Analisis Manual")
    selected_tickers = st.multiselect(
        "Centang saham yang ingin dianalisis:",
        options=idx_tickers,
        default=["BBCA.JK"],
        help="Kamu bisa memilih lebih dari satu saham sekaligus"
    )

    st.markdown(
        "<small style='color:gray'>Mode ini menganalisis satu atau beberapa saham dengan tampilan grafik dan indikator lengkap (teknikal, fundamental, dan makroekonomi).</small>",
        unsafe_allow_html=True,
    )
analyzer = AutomatedStockAnalyzer()

# Utilities
def score_to_color(score):
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    elif score >= 40:
        return "⚪"
    else:
        return "🔴"

def style_action(val):
    if isinstance(val, str):
        v = val.lower()
        if "buy" in v or "enter" in v:
            return "background-color: #16a34a; color: black"
        if "sell" in v or "short" in v:
            return "background-color: #dc2626; color: white"
        if "hold" in v:
            return "background-color: #f59e0b; color: black"
    return ""

# ---------------------------
# Generate recommendations (run)
# ---------------------------
# run_button = st.button("🚀 Generate Recommendations")
# if run_button:
#     with st.spinner("Analysing... This may take a while for many tickers."):
#         res_pack = analyzer.generate_recommendations(
#             tickers=tickers,
#             period=period_choice,
#             interval=interval_choice,
#             top_n=top_n,
#             # capital=capital,
#             risk_percent=risk_percent
#         )
#         # keep in session for refresh / watchlist
#         st.session_state["last_results"] = res_pack
#         st.session_state["last_run"] = datetime.now().isoformat()
#         st.rerun()

if analysis_mode == "All Analysis (Full Auto)":
    run_button = st.button("🚀 Generate Recommendations")
    if run_button:
        with st.spinner("Analysing... This may take a while for many tickers."):
            res_pack = analyzer.generate_recommendations(
                tickers=tickers,
                period=period_choice,
                interval=interval_choice,
                top_n=top_n,
                # capital=capital,
                risk_percent=risk_percent
            )
            st.session_state["last_results"] = res_pack
            st.session_state["last_run"] = datetime.now().isoformat()
            st.rerun()

elif analysis_mode == "Single Analysis (Manual)":
    run_button = st.button("🔍 Analyze Selected Stock")
    if run_button and selected_tickers:
        with st.spinner(f"Analysing {len(selected_tickers)} stock(s)..."):
            res_pack = analyzer.generate_recommendations(
                tickers=selected_tickers,
                period=period_choice,
                interval=interval_choice,
                top_n=top_n,
                # capital=capital,
                risk_percent=risk_percent
            )
            st.session_state["last_results"] = res_pack
            st.session_state["last_run"] = datetime.now().isoformat()
            st.rerun()

# auto refresh simple (non-blocking)
if auto_refresh and "last_results" in st.session_state:
    import time
    last_run = datetime.fromisoformat(st.session_state.get("last_run"))
    if (datetime.now() - last_run).total_seconds() > refresh_interval:
        st.rerun()

# Load last results if available
res_pack = st.session_state.get("last_results", None)

if res_pack is None:
    st.info("Belum ada hasil analisis. Klik 'Generate Recommendations' untuk memulai menganalisa.")
    st.stop()

# ---------------------------
# Top recommendations table
# ---------------------------
ranked = res_pack.get("top", []) 
st.subheader("Top Recommendations Today")
    # list of (ticker, score)
if not ranked:
    st.warning("Tidak ada rekomendasi teratas (coba run ulang atau ganti tickers).")
else:
    rows = []
    for i, (ticker, score) in enumerate(ranked, start=1):
        analysis = res_pack["results"].get(ticker)
        if analysis:
            fm = analysis["fundamental_metrics"]
            last_price = analysis["price_data"]["Close"].iloc[-1]
            change_pct = None

            # Ambil nama perusahaan
            try:
                info = yf.Ticker(ticker).info
                company_name = info.get("longName", "Unknown")
            except Exception:
                company_name = "Unknown"

            if len(analysis["price_data"]) >= 2:
                prev = analysis["price_data"]["Close"].iloc[-2]
                change_pct = (last_price - prev) / prev * 100 if prev != 0 else 0

            # Tambahkan ke tabel utama
            rows.append({
                "No": i, 
                "Ticker": f"{ticker}\n - {company_name}",
                "Price": round(float(last_price), 2) if pd.notna(last_price) else None,
                "Change%": round(float(change_pct), 2) if change_pct is not None else None,
                "Signal": res_pack["results"][ticker]["technical_score"]["signal"],
                "Score": round(res_pack["results"][ticker]["technical_score"]["score"], 1),
                "Trend": "⬆️" if res_pack["results"][ticker]["technical_score"]["components"]["ma_trend_score"] > 50 else "↔️" if 40 <= res_pack["results"][ticker]["technical_score"]["components"]["ma_trend_score"] <= 60 else "⬇️",
            })

    top_df = pd.DataFrame(rows).reset_index(drop=True)
    # style the Signal column
    def color_signal(v):
        if "BUY" in v:
            return "background-color: #16a34a; color: white"
        if "HOLD" in v:
            return "background-color: #f59e0b; color: white"
        if "SELL" in v:
            return "background-color: #dc2626; color: black"
        return ""

    top_styled = top_df.style.applymap(color_signal, subset=["Signal"])
    # st.dataframe(top_styled, width="stretch")
    st.dataframe(top_styled, width="stretch", hide_index=True)

# ---------------------------
# Expandable detail panels
# ---------------------------
st.subheader("Details (expand a row to view full analysis)")
for ticker, score in ranked:
    analysis = res_pack["results"].get(ticker)

    # Ambil nama perusahaan dari metadata
    info = yf.Ticker(ticker).info
    company_name = info.get("longName", "Unknown Company")
    
    if not analysis:
        continue

    # accordion-like
    with st.expander(f"{ticker} ({company_name}) — Score: {analysis['technical_score']['score']} — Signal: {analysis['technical_score']['signal']}"):
        cols = st.columns([2, 2, 1])
        # left: mini chart
        with cols[0]:
            st.markdown("**Candlestick (mini)**")
            
            df = analysis["price_data"].copy().tail(120)
            if not df.empty:
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Price"
                    )
                ])

                # Tambahkan overlay moving averages jika tersedia
                for ma in ["MA_5", "MA_20", "MA_50", "EMA_8", "EMA_21"]:
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df[ma],
                            name=ma,
                            line=dict(width=1)
                        ))

                # Layout tampilan chart
                fig.update_layout(
                    template="plotly_dark",
                    height=360,
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=True,
                    width=None  # biarkan Streamlit atur otomatis
                )

                # ✅ Versi terbaru Streamlit (tanpa warning)
                st.plotly_chart(
                    fig,
                    config={
                        "displaylogo": False,
                        "responsive": True,
                        "modeBarButtonsToRemove": [
                            "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
                        ]
                    },
                    use_container_width=True  # masih aman di Streamlit <1.38, nanti ganti ke width='stretch'
                )
            else:
                st.write("No price data to show.")

        # middle: indicators summary
        with cols[1]:
            st.markdown("**Indicators Summary**")
            ta = analysis["price_data"].iloc[-1].to_dict()
            comp = analysis["technical_score"].get("components", {})
            fm = analysis["fundamental_metrics"]
            # show useful metrics
            items = {
                "Close": ta.get("Close"),
                "RSI_14": round(ta.get("RSI_14", 0), 2) if ta.get("RSI_14") is not None else None,
                "MACD_Hist": round(ta.get("MACD_Hist", 4), 4) if ta.get("MACD_Hist") is not None else None,
                "Vol_Ratio": round(ta.get("Vol_Ratio", 1), 2) if ta.get("Vol_Ratio") is not None else None,
                "ATR_14": round(ta.get("ATR_14", 0), 4),
                "EMA8>EMA21": bool(ta.get("EMA8_above_EMA21", 0)),
            }
            kd = pd.DataFrame(list(items.items()), columns=["Metric", "Value"])
            st.dataframe(kd, width="stretch")

            st.markdown("**Fundamental (quick)**")
            fm_quick = {
                "Price": fm.get("current_price"),
                "P/E": fm.get("pe_ratio"),
                "P/B": fm.get("pb_ratio"),
                "ROE": fm.get("roe"),
                "DER": fm.get("der"),
            }
            st.dataframe(pd.DataFrame(list(fm_quick.items()), columns=["Metric", "Value"]), width="stretch")

        # right: 3-day plan + actions
        with cols[2]:
            st.markdown("**3-Day Plan**")
            plan = analysis["3day_plan"]
            st.write(f"Action: **{plan['action']}**")
            st.write(f"Entry: {plan['entry']}")
            st.write(f"Stop: {plan['stop']}")
            st.write(f"Target: {plan['target']}")
            st.write(f"Position size: {plan['position_size']}  (Value: {plan['position_value']})")
            st.markdown("---")
            if st.button(f"➕ Add {ticker} to Watchlist", key=f"wl_{ticker}"):
                if ticker not in st.session_state["watchlist"]:
                    st.session_state["watchlist"].append(ticker)
                    st.success(f"{ticker} added to watchlist")
                else:
                    st.info(f"{ticker} is already in watchlist")

        # AI summary (simple templated)
        st.markdown("**AI Summary (templated)**")
        score_val = analysis["technical_score"]["score"]
        sig = analysis["technical_score"]["signal"]
        rsi = analysis["price_data"].iloc[-1].get("RSI_14", None)
        summary = f"{ticker} shows a {sig.lower()} bias with score {score_val:.1f}/100. RSI ~ {rsi:.1f} (if available). Consider entry near {plan['entry']} with stop {plan['stop']} and target {plan['target']} (R=2)."
        st.info(summary)

# ---------------------------
# Watchlist panel
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")
wl = st.session_state.get("watchlist", [])
st.sidebar.write("Saved tickers:")
for t in wl:
    st.sidebar.write(f"- {t}")
if st.sidebar.button("Export Watchlist CSV"):
    if wl:
        wl_df = pd.DataFrame(wl, columns=["Ticker"])
        csv_bytes = wl_df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Download watchlist", data=csv_bytes, file_name="watchlist.csv", mime="text/csv")
    else:
        st.sidebar.info("Watchlist kosong.")

# ---------------------------
# Export full results
# ---------------------------
st.markdown("---")
if st.button("⬇️ Download full results (CSV)"):
    # create combined CSV
    rows = []
    for t, analysis in res_pack["results"].items():
        last = analysis["price_data"].iloc[-1]
        plan = analysis["3day_plan"]
        rows.append({
            "ticker": t,
            "close": last.get("Close"),
            "rsi": last.get("RSI_14"),
            "macd_hist": last.get("MACD_Hist"),
            "score": analysis["technical_score"]["score"],
            "signal": analysis["technical_score"]["signal"],
            "action": plan["action"],
            "entry": plan["entry"],
            "stop": plan["stop"],
            "target": plan["target"],
        })
    out_df = pd.DataFrame(rows)
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")

# Macro summary
st.sidebar.markdown("---")
st.sidebar.subheader("Macro Context")
macro = res_pack.get("macro", {})
st.sidebar.write(f"Market mood: **{macro.get('market_sentiment','N/A')}**")
for k, v in macro.get("macros", {}).items():
    st.sidebar.write(f"{k}: {v.get('last', 'N/A')} ({round(v.get('pct',0),2)}%)")

st.caption("Disclaimer: This tool uses yfinance which is not a guaranteed real-time feed. Use recommendations as decision support, not as automated execution signals.")
