import yfinance as yf
import streamlit as st
import pandas as pd
from analyzer_full import AutomatedStockAnalyzer
from datetime import datetime
import plotly.graph_objects as go
from idx_listed import get_idx_tickers
from pathlib import Path

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Automated Stock Recommender", layout="wide", page_icon="📈")

# ---------------------------
# CSS Styling - Modern Sidebar
# ---------------------------
st.markdown("""
    <style>
    body, .stApp {background-color:#0E1117; color:#E2E8F0;}
    h1,h2,h3,h4,h5,h6 {color:#F8FAFC !important; font-weight:600;}
    .stMarkdown {color:#CBD5E1 !important;}
    section[data-testid="stSidebar"] {
        background-color:#111827;
        padding: 1.2rem 1rem;
        border-right: 1px solid #1F2937;
    }
    .sidebar-box {
        background-color:#1E293B;
        border:1px solid #334155;
        border-radius:12px;
        padding:0.8rem 0.9rem;
        margin-bottom:1rem;
    }
    button[kind="primary"], div.stButton > button {
        background: linear-gradient(90deg, #FACC15, #FB923C);
        color:#111827 !important;
        border:none;
        border-radius:8px;
        padding:0.5rem 0.8rem;
        font-weight:600;
        transition:0.2s;
    }
    button[kind="primary"]:hover, div.stButton > button:hover {
        background: linear-gradient(90deg, #FB923C, #FACC15);
        transform: translateY(-1px);
    }
    hr, .stDivider {border-color:#1F2937; margin:0.8rem 0;}
    label[data-testid="stMarkdownContainer"] {
        color:#E2E8F0 !important;
        font-size:0.9rem !important;
    }
    .sidebar-watchlist {
        background-color:#1E293B;
        border-radius:10px;
        padding:0.7rem;
        border:1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------
# Fungsi baca sektor saham
# ---------------------------
@st.cache_data(show_spinner=False)
def read_sector_tickers(excel_path: Path):
    """Baca kolom kode saham dari file sektor sesuai format idx_listed.py"""
    if not excel_path.exists():
        st.error(f"❌ File tidak ditemukan: {excel_path}")
        return []
    try:
        df = pd.read_excel(excel_path)
        possible_cols = [c for c in df.columns if "Code" in c or "Kode" in c or "Emiten" in c]
        if not possible_cols:
            st.warning(f"⚠️ Tidak ditemukan kolom kode di {excel_path.name}")
            return []
        col = possible_cols[0]
        tickers = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"[^A-Za-z0-9]", "", regex=True)
            .tolist()
        )
        tickers = [f"{t}.JK" for t in tickers if len(t) <= 6]
        return sorted(list(set(tickers)))
    except Exception as e:
        st.error(f"❌ Gagal membaca file Excel sektor {excel_path.name}: {e}")
        return []


# ---------------------------
# App header
# ---------------------------
st.title("📈 Automated Stock Recommender")
st.markdown("Real-time-ish recommendation engine (yfinance). Dashboard shows top recommended stocks and expandable detail per stock.")


# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("🛠️ Controls")
    st.markdown("Configure your analysis below:")

    # === Pilihan sektor saham ===
    st.markdown("### 📊 Pilih Sektor Saham")
    stock_folder = Path("stock")
    sector_files = {
        "All Stocks (IDX)": None,
        "Basic Materials": stock_folder / "Basic_Materials.xlsx",
        "Energy": stock_folder / "Energy.xlsx",
        "Industrials": stock_folder / "Industrials.xlsx",
        "Infrastructures": stock_folder / "Infrastructures.xlsx",
        "Properties & Real Estate": stock_folder / "Properties_Real_Estate.xlsx",
        "Technology": stock_folder / "Technology.xlsx",
        "Transportation & Logistic": stock_folder / "Transportation_Logistic.xlsx",
        "Consumer Non-Cyclicals": stock_folder / "Consumer_Non_Cyclicals.xlsx",
    }

    selected_sector = st.selectbox("Pilih sektor:", list(sector_files.keys()), index=0)

    if selected_sector == "All Stocks (IDX)":
        try:
            tickers_list = get_idx_tickers()
            default_list = ", ".join(tickers_list)
            st.success(f"✅ Loaded {len(tickers_list)} tickers dari IDX")
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat IDX otomatis: {e}")
            default_list = ""
    else:
        sector_file = sector_files[selected_sector]
        if sector_file and sector_file.exists():
            tickers_list = read_sector_tickers(sector_file)
            default_list = ", ".join(tickers_list)
            st.success(f"✅ Loaded {len(tickers_list)} tickers dari sektor {selected_sector}")
        else:
            st.warning(f"⚠️ File sektor {selected_sector} belum ditemukan.")
            default_list = ""

    tickers = [t.strip().upper() for t in default_list.split(",") if t.strip()]

    # Time & analysis settings
    st.markdown("### ⏱️ Time Settings")
    period_choice = st.selectbox("🗓️ Period for analysis", ["1mo", "3mo", "6mo", "1y"], index=0)
    interval_choice = st.selectbox("⏰ Interval (yfinance)", ["60m", "90m", "1d"], index=0)

    # Analysis settings
    st.markdown("### 📊 Analysis Settings")
    top_n = st.number_input("🏅 Top N recommendations", min_value=1, max_value=50, value=5, step=1)
    capital = st.number_input("💰 Capital (currency)", value=100000000, step=1000000)
    risk_percent = st.slider("⚠️ Risk per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    # Auto refresh
    st.markdown("### 🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable auto refresh", value=False)
    refresh_interval = st.number_input("⏳ Interval (seconds)", min_value=30, max_value=3600, value=600, step=10)

    st.markdown("---")
    if st.button("🔃 Refresh now"):
        st.rerun()


# ---------------------------
# Watchlist session
# ---------------------------
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []


# ---------------------------
# Mode Analisis
# ---------------------------
st.markdown("### ⚙️ Pilih Mode Analisis")
analysis_mode = st.radio("Mode analisis:", ["All Analysis (Full Auto)", "Single Analysis (Manual)"], index=0, horizontal=True)

selected_tickers = []
if analysis_mode == "Single Analysis (Manual)":
    selected_tickers = st.multiselect("Centang saham yang ingin dianalisis:", options=tickers, default=tickers[:1] if tickers else [])
    st.markdown("<small style='color:gray'>Mode ini menganalisis satu atau beberapa saham dengan tampilan grafik dan indikator lengkap.</small>", unsafe_allow_html=True)

analyzer = AutomatedStockAnalyzer()


# ---------------------------
# Generate Recommendations
# ---------------------------
if analysis_mode == "All Analysis (Full Auto)":
    if st.button("🚀 Generate Recommendations"):
        with st.spinner("Analysing..."):
            res_pack = analyzer.generate_recommendations(tickers=tickers, period=period_choice, interval=interval_choice, top_n=top_n, risk_percent=risk_percent)
            st.session_state["last_results"] = res_pack
            st.session_state["last_run"] = datetime.now().isoformat()
            st.rerun()
elif analysis_mode == "Single Analysis (Manual)" and selected_tickers:
    if st.button("🔍 Analyze Selected Stock"):
        with st.spinner(f"Analysing {len(selected_tickers)} stock(s)..."):
            res_pack = analyzer.generate_recommendations(tickers=selected_tickers, period=period_choice, interval=interval_choice, top_n=top_n, risk_percent=risk_percent)
            st.session_state["last_results"] = res_pack
            st.session_state["last_run"] = datetime.now().isoformat()
            st.rerun()


# ---------------------------
# Auto refresh
# ---------------------------
if auto_refresh and "last_results" in st.session_state:
    last_run = datetime.fromisoformat(st.session_state.get("last_run"))
    if (datetime.now() - last_run).total_seconds() > refresh_interval:
        st.rerun()


# ---------------------------
# Hasil Analisis
# ---------------------------
res_pack = st.session_state.get("last_results", None)
if res_pack is None:
    st.info("Belum ada hasil analisis. Klik 'Generate Recommendations' untuk memulai.")
    st.stop()


# ---------------------------
# Top Recommendations
# ---------------------------
ranked = res_pack.get("top", [])
st.subheader("Top Recommendations Today")

if not ranked:
    st.warning("Tidak ada rekomendasi teratas.")
else:
    rows = []
    for i, (ticker, score) in enumerate(ranked, start=1):
        analysis = res_pack["results"].get(ticker)
        if analysis:
            fm = analysis["fundamental_metrics"]
            last_price = analysis["price_data"]["Close"].iloc[-1]
            change_pct = None
            try:
                info = yf.Ticker(ticker).info
                company_name = info.get("longName", "Unknown")
            except Exception:
                company_name = "Unknown"
            if len(analysis["price_data"]) >= 2:
                prev = analysis["price_data"]["Close"].iloc[-2]
                change_pct = (last_price - prev) / prev * 100 if prev != 0 else 0
            rows.append({
                "No": i,
                "Ticker": f"{ticker}\n - {company_name}",
                "Price": round(float(last_price), 2),
                "Change%": round(float(change_pct), 2) if change_pct else None,
                "Signal": analysis["technical_score"]["signal"],
                "Score": round(analysis["technical_score"]["score"], 1),
                "Trend": "⬆️" if analysis["technical_score"]["components"]["ma_trend_score"] > 50 else "↔️" if 40 <= analysis["technical_score"]["components"]["ma_trend_score"] <= 60 else "⬇️",
            })
    top_df = pd.DataFrame(rows).reset_index(drop=True)

    def color_signal(v):
        if "BUY" in v: return "background-color:#16a34a;color:white"
        if "HOLD" in v: return "background-color:#f59e0b;color:white"
        if "SELL" in v: return "background-color:#dc2626;color:black"
        return ""

    st.dataframe(top_df.style.applymap(color_signal, subset=["Signal"]), width="stretch", hide_index=True)


# ---------------------------
# Detail Panels
# ---------------------------
st.subheader("Details (expand a row to view full analysis)")
for ticker, score in ranked:
    analysis = res_pack["results"].get(ticker)
    if not analysis:
        continue

    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName", "Unknown Company")
    except Exception:
        company_name = "Unknown Company"

    signal_text = analysis["technical_score"]["signal"]
    signal_upper = signal_text.upper()
    trend_score = analysis["technical_score"]["components"].get("ma_trend_score", 50)
    if "BUY" in signal_upper:
        signal_emoji = "🟢"
    elif "SELL" in signal_upper:
        signal_emoji = "🔴"
    elif "HOLD" in signal_upper:
        signal_emoji = "🟡"
    else:
        signal_emoji = "⚪"

    emoji_trend = "⬆️" if trend_score > 60 else "⬇️" if trend_score < 40 else "↔️"
    trend_text = "Uptrend" if trend_score > 60 else "Downtrend" if trend_score < 40 else "Sideway"

    expander_label = f"{signal_emoji} {ticker} ({company_name}) — Score: {analysis['technical_score']['score']} — Signal: {signal_text} {emoji_trend} {trend_text}"

    with st.expander(expander_label, expanded=False):
        cols = st.columns([2, 2, 1])
        df = analysis["price_data"].copy().tail(120)

        with cols[0]:
            st.markdown("**Candlestick (mini)**")
            if not df.empty:
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
                )])

                support = analysis.get("support")
                resistance = analysis.get("resistance")
                if support:
                    fig.add_hline(y=support, line_dash="dot", line_color="green",
                                  annotation_text=f"Support {support}", annotation_position="bottom right")
                if resistance:
                    fig.add_hline(y=resistance, line_dash="dot", line_color="red",
                                  annotation_text=f"Resistance {resistance}", annotation_position="top right")

                for ma in ["MA_5", "MA_20", "MA_50", "EMA_8", "EMA_21"]:
                    if ma in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1)))

                fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
                st.plotly_chart(fig, config={"displaylogo": False, "responsive": True}, use_container_width=True)
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
            st.markdown("**3-Day Plan**")
            plan = analysis["3day_plan"]
            st.write(f"Action: **{plan['action']}**")
            st.write(f"Entry: {plan['entry']}")
            st.write(f"Stop: {plan['stop']}")
            st.write(f"Target: {plan['target']}")
            st.write(f"Position size: {plan['position_size']} (Value: {plan['position_value']})")

            st.markdown("---")
            st.markdown("⚡ Support–Resistance Timing")
            st.write(f"Nearest Support: **{analysis.get('support', 'N/A')}**")
            st.write(f"Nearest Resistance: **{analysis.get('resistance', 'N/A')}**")
            timing_advice = analysis.get("timing_advice", None)
            if timing_advice:
                st.info(timing_advice)
            else:
                st.info("⏸ Tidak ada sinyal timing signifikan saat ini.")

            st.markdown("---")
            if st.button(f"➕ Add {ticker} to Watchlist", key=f"wl_{ticker}"):
                if ticker not in st.session_state["watchlist"]:
                    st.session_state["watchlist"].append(ticker)
                    st.success(f"{ticker} added to watchlist")
                else:
                    st.info(f"{ticker} is already in watchlist")

        st.markdown("**AI Summary (templated)**")
        score_val = analysis["technical_score"]["score"]
        sig = analysis["technical_score"]["signal"]
        rsi = analysis["price_data"].iloc[-1].get("RSI_14", None)
        summary = (
            f"{ticker} shows a {sig.lower()} bias with score {score_val:.1f}/100. "
            f"RSI ~ {rsi:.1f} (if available). "
            f"Support {analysis.get('support', '-')} , Resistance {analysis.get('resistance', '-')}. "
            f"{analysis.get('timing_advice', '')} "
            f"Consider entry near {plan['entry']} with stop {plan['stop']} and target {plan['target']} (R=2)."
        )
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


# ---------------------------
# Macro summary
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Macro Context")
macro = res_pack.get("macro", {})
st.sidebar.write(f"Market mood: **{macro.get('market_sentiment','N/A')}**")
for k, v in macro.get("macros", {}).items():
    st.sidebar.write(f"{k}: {v.get('last', 'N/A')} ({round(v.get('pct',0),2)}%)")

st.caption("Disclaimer: This tool uses yfinance which is not a guaranteed real-time feed.")
