import os
import yfinance as yf
import streamlit as st
import pandas as pd
from analyzer_full import AutomatedStockAnalyzer
from datetime import datetime
import plotly.graph_objects as go
from idx_listed import get_idx_tickers
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from analyzer_full_one_day import AutomatedStockAnalyzerDailyTP, render_daily_tp_page
import textwrap
import json
from datetime import datetime


load_dotenv()
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK ")

# ---------------------------
# Winrate calculation functions
# ---------------------------
def load_history():
    path = "trade_history/signal_history.json"
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # Jika file rusak → reset
        with open(path, "w") as f:
            json.dump({}, f, indent=4)
        return {}

def calculate_winrate(data):
    total = 0
    wins = 0

    for ticker, records in data.items():
        for r in records:
            total += 1
            if r["result"] == "WIN":
                wins += 1

    if total == 0:
        return 0, 0, 0

    return (wins / total * 100), wins, total


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
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("🛠️ Controls")
    st.markdown("Configure your analysis below:")

    # ===== App page selector =====
    app_page = st.selectbox("Menu", ["Dashboard","Simulator Investasi", "Analisa TP Harian", "Winrate Mingguan"], index=0)

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
    period_choice = st.selectbox("🗓️ Period for analysis", ["1mo", "3mo", "6mo", "5d","1y"], index=0)
    interval_choice = st.selectbox("⏰ Interval (yfinance)", ["1h","30m","60m", "90m", "1d"], index=0)

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

analyzer = AutomatedStockAnalyzer()
analyzer.set_watchlist(st.session_state["watchlist"])
analyzer.set_discord_notifier(DISCORD_WEBHOOK, st.session_state.get("watchlist", []))

# --- Simulator Investasi page
try:
    current_app_page = app_page
except NameError:
    current_app_page = "Dashboard"

if current_app_page == "Simulator Investasi":
    st.title("💼 Simulator Investasi Jangka Panjang")
    st.markdown("Gunakan simulator ini untuk memproyeksikan pertumbuhan investasi dengan analisis Monte Carlo, multi-analis, dan sensitivitas.")

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        sim_ticker = st.text_input("Ticker (misal: BBCA.JK)", value="BBCA.JK")
        sim_years = st.number_input("Horizon (tahun)", 1, 50, 10, 1)
        sim_initial = st.number_input("Modal Awal (Rp)", min_value=0.0, value=10_000_000.0, step=100_000.0, format="%.2f")
    with col2:
        sim_contrib = st.number_input("Kontribusi Periodik (Rp)", min_value=0.0, value=0.0, step=100_000.0, format="%.2f")
        sim_freq_choice = st.selectbox("Frekuensi Kontribusi", ["Bulanan (12)", "Mingguan (52)", "Tahunan (1)"], index=0)
        sim_freq = 12 if "Bulanan" in sim_freq_choice else (52 if "Mingguan" in sim_freq_choice else 1)
    with col3:
        sim_method = st.radio("Metode", ["Monte Carlo", "Deterministik"], horizontal=True)
        sim_n = st.slider("Jumlah Simulasi (n)", 100, 5000, 1000, step=100)
        use_bootstrap = st.checkbox("Gunakan Bootstrap (lebih realistis)", value=True)

    st.markdown("Masukkan estimasi return analis tahunan (dalam %) — pisahkan dengan koma. \
                Tambahkan bobot opsional dengan format `return:bobot`, misal: `7:0.6,10:0.4`")
    analyst_input = st.text_input("Estimasi Analis", value="7,10")

    if st.button("▶️ Jalankan Simulasi"):
        analyst_list = []
        for token in analyst_input.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                if ":" in token:
                    r_str, w_str = token.split(":")
                    analyst_list.append({"r": float(r_str) / 100.0, "w": float(w_str)})
                else:
                    analyst_list.append(float(token) / 100.0)
            except Exception:
                st.warning(f"Format salah pada token '{token}'. Abaikan.")
                continue

        with st.spinner("⏳ Menjalankan simulasi..."):
            sim_result = analyzer.simulate_long_term_investment(
                ticker=sim_ticker.strip().upper(),
                years=sim_years,
                initial_capital=sim_initial,
                periodic_contribution=sim_contrib,
                contribution_freq_per_year=sim_freq,
                n_simulations=sim_n,
                method="deterministic" if sim_method.startswith("Det") else "mc",
                analyst_estimates=analyst_list,
                use_bootstrap=use_bootstrap,
                seed=42
            )

        if "error" in sim_result:
            st.error(f"❌ Simulasi gagal: {sim_result['error']}")
        else:
            # ========== RINGKASAN ==========
            st.subheader("📊 Ringkasan Hasil")
            colA, colB, colC = st.columns(3)
            colA.metric("Rata-rata (Mean)", f"{sim_result.get('mc_mean', sim_result.get('projection', 0)):,.0f}")
            colB.metric("Median (P50)", f"{sim_result.get('mc_percentiles', {}).get(50, 0):,.0f}")
            colC.metric("Sharpe Ratio", f"{sim_result.get('sharpe_ratio', 0):.2f}")

            colD, colE, colF = st.columns(3)
            colD.metric("Probabilitas Rugi", f"{sim_result.get('prob_loss', 0)*100:.1f}%")
            colE.metric("Expected CAGR", f"{sim_result.get('expected_cagr', sim_result.get('historical_cagr', 0))*100:.2f}%")
            colF.metric("Simulasi", f"{sim_result.get('n_simulations', 0)}x")

            # ========== DISTRIBUSI ==========
            if sim_result["method"] == "mc":
                st.subheader("📈 Distribusi Hasil Monte Carlo")
                df_percentiles = pd.DataFrame.from_dict(sim_result["mc_percentiles"], orient="index", columns=["Final Value"])
                df_percentiles.index = [f"P{int(x)}" for x in df_percentiles.index]
                st.dataframe(df_percentiles.style.format("{:,.0f}"))

                fig = go.Figure()
                fig.add_trace(go.Box(y=np.array(list(sim_result["mc_percentiles"].values())),
                                     name="Percentile Spread", boxmean=True))
                fig.update_layout(title="Distribusi Nilai Akhir (Monte Carlo)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # ========== SAMPLE PATHS ==========
                sample_paths = sim_result.get("sample_paths")
                if sample_paths:
                    st.subheader("📉 Jalur Simulasi (Sample Paths)")
                    fig2 = go.Figure()
                    for p in sample_paths[:min(50, len(sample_paths))]:
                        fig2.add_trace(go.Scatter(y=p, mode="lines", line=dict(width=1), opacity=0.6))
                    fig2.update_layout(title="Simulasi Investasi Jangka Panjang", template="plotly_dark", height=400)
                    st.plotly_chart(fig2, use_container_width=True)

                # ========== SENSITIVITY ==========
                if sim_result.get("sensitivity"):
                    st.subheader("📊 Analisis Sensitivitas")
                    sens_df = pd.DataFrame.from_dict(sim_result["sensitivity"], orient="index", columns=["Nilai"])
                    st.dataframe(sens_df.style.format("{:,.0f}"))

            else:
                st.subheader("💡 Proyeksi Deterministik")
                st.write(f"**CAGR Historis:** {sim_result.get('historical_cagr', 0)*100:.2f}%")
                st.write(f"**Proyeksi Nilai Akhir:** {sim_result.get('projection', 0):,.0f}")

            # ========== ANALYST SCENARIOS ==========
            if sim_result.get("analyst_scenarios"):
                st.subheader("👥 Estimasi Analis")
                df_analyst = pd.DataFrame(sim_result["analyst_scenarios"])
                df_analyst["annual_return_%"] = df_analyst["annual_return"] * 100
                st.dataframe(df_analyst[["annual_return_%", "weight", "final_value"]].style.format("{:,.2f}"))
    st.stop()

if current_app_page == "Analisa TP Harian":
    analyzer_tp = AutomatedStockAnalyzerDailyTP()
    analyzer_tp.set_watchlist(st.session_state.get("watchlist", []))
    render_daily_tp_page(analyzer_tp, tickers, period_choice, interval_choice, top_n, capital, risk_percent)
    st.stop()


if current_app_page == "Winrate Mingguan":
    st.title("📊 Winrate Mingguan — AI Tracker")

    data = load_history()

    winrate, wins, total = calculate_winrate(data)

    st.metric("AI Total Winrate", f"{winrate:.2f}%")
    st.write(f"Total WIN: {wins}")
    st.write(f"Total Sinyal: {total}")

    st.markdown("---")
    st.subheader("🎯 Riwayat Sinyal")

    for ticker, records in data.items():
        st.markdown(f"### 📌 {ticker}")
        st.dataframe(pd.DataFrame(records))

    st.stop()



# ---------------------------
# App header
# ---------------------------
st.title("📈 Automated Stock Recommender")
st.markdown("Real-time-ish recommendation engine (yfinance). Dashboard shows top recommended stocks and expandable detail per stock.")

# ---------------------------
# Mode Analisis
# ---------------------------
st.markdown("### ⚙️ Pilih Mode Analisis")
analysis_mode = st.radio("Mode analisis:", ["All Analysis (Full Auto)", "Single Analysis (Manual)"], index=0, horizontal=True)

selected_tickers = []
if analysis_mode == "Single Analysis (Manual)":
    selected_tickers = st.multiselect("Centang saham yang ingin dianalisis:", options=tickers, default=tickers[:1] if tickers else [])
    st.markdown("<small style='color:gray'>Mode ini menganalisis satu atau beberapa saham dengan tampilan grafik dan indikator lengkap.</small>", unsafe_allow_html=True)

# ---------------------------
# Generate Recommendations
# ---------------------------
if analysis_mode == "All Analysis (Full Auto)":
    if st.button("🚀 Generate Recommendations"):
        with st.spinner("Analysing..."):
            res_pack = analyzer.generate_recommendations(
                tickers=tickers,
                period=period_choice,
                interval=interval_choice,
                top_n=top_n,
                risk_percent=risk_percent,
                capital=capital,
                mode="deep"
            )

            st.session_state["last_results"] = res_pack
            st.session_state["last_run"] = datetime.now().isoformat()
            st.rerun()
elif analysis_mode == "Single Analysis (Manual)" and selected_tickers:
    if st.button("🔍 Analyze Selected Stock"):
        with st.spinner(f"Analysing {len(selected_tickers)} stock(s)..."):
            res_pack = analyzer.generate_recommendations(
                tickers=selected_tickers,
                period=period_choice,
                interval=interval_choice,
                top_n=top_n,
                risk_percent=risk_percent,
                capital=capital,
                mode="deep"
            )

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


ranked = res_pack.get("top", [])
def render_html(html: str):
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# ---------------------------
# Top Recommendations
# ---------------------------

# Fungsi get_quick_prediction yang lebih baik:
def get_quick_prediction(ticker: str, analyzer: AutomatedStockAnalyzer) -> str:
    """Prediksi cepat untuk tampilan tabel"""
    try:
        pred = analyzer.predict_price_direction(ticker, forecast_days=5)
        if "error" in pred:
            return "N/A"
        
        up_days = pred["up_days"]
        down_days = pred["down_days"]
        direction = pred["net_direction"]
        
        if direction == "UP":
            return f"📈 {up_days}/{down_days}H"
        elif direction == "DOWN":
            return f"📉 {up_days}/{down_days}H"
        else:
            return f"↔️ {up_days}/{down_days}H"
    except Exception as e:
        return "N/A"
    
st.subheader("Top Recommendations Today")

if not ranked:
    st.warning("Tidak ada rekomendasi teratas.")
else:
    rows = []
    for i, item in enumerate(ranked, start=1):
        if isinstance(item, dict):
            ticker = item.get("ticker")
            score = item.get("score")
            projects = item.get("projects")
        elif isinstance(item, (list, tuple)):
            if len(item) == 3:
                ticker, score, projects = item
            elif len(item) == 2:
                ticker, score = item
                projects = 0  # default
            else:
                st.error(f"Struktur data tidak dikenali: {item}")
                continue
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
                "Prediksi 5H": f"{get_quick_prediction(ticker, analyzer)}",
                "Trend": "⬆️" if analysis["technical_score"]["components"]["ma_trend_score"] > 50 else "↔️" if 40 <= analysis["technical_score"]["components"]["ma_trend_score"] <= 60 else "⬇️",
                "Demand": analysis["demand_analysis"].get("demand_score", 0),  # Perhatikan key yang benar
                "Projects": projects,  # Gunakan nilai projects dari tuple
            })
    top_df = pd.DataFrame(rows).reset_index(drop=True)

   
    def color_signal(v):
        if "BUY" in v: return "background-color:#16a34a;color:white"
        if "HOLD" in v: return "background-color:#f59e0b;color:white"
        if "SELL" in v: return "background-color:#dc2626;color:black"
        return ""

    st.dataframe(top_df.style.map(color_signal, subset=["Signal"]), width="stretch", hide_index=True)

# ---------------------------
# Detail Panels
# ---------------------------
st.subheader("Details (expand a row to view full analysis)")
# PERBAIKAN: unpack 3 nilai
for item in ranked:
    # HANDLE BERBAGAI STRUKTUR DATA
    if isinstance(item, dict):
        ticker = item.get("ticker")
    elif isinstance(item, (list, tuple)):
        ticker = item[0] if len(item) > 0 else None
    elif isinstance(item, str):
        ticker = item
    else:
        continue
    
    if not ticker:
        continue
        
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
        cols = st.columns([4, 2])
        df = analysis["price_data"].copy().tail(120)

        with cols[0]:
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

            # Backtesting section dengan error handling
            st.markdown("⚡ **Backtesting (60m candles)**")
            try:
                res = analyzer.backtest_daily_tp(
                    ticker=ticker,
                    period="6mo",
                    interval="60m",
                    tp_pct=0.0075,
                    sl_pct=0.004,
                    horizon_candles=12
                )
                
                if isinstance(res, dict) and 'summary' in res:
                    summary = res['summary']
                    
                    # Tampilkan metrics yang lebih jelas
                    col_bt1, col_bt2, col_bt3 = st.columns(3)
                    
                    with col_bt1:
                        st.metric("Total Trades", summary.get('total_trades', 0))
                    
                    with col_bt2:
                        winrate = summary.get('winrate_pct', 0)
                        winrate_color = "normal" if winrate >= 50 else "inverse"
                        st.metric("Winrate", f"{winrate:.1f}%", delta_color=winrate_color)
                    
                    with col_bt3:
                        avg_pnl = summary.get('avg_pnl_pct', 0)
                        pnl_color = "normal" if avg_pnl > 0 else "inverse"
                        st.metric("Avg PnL", f"{avg_pnl:.3f}%", delta_color=pnl_color)
                    
                    # Info tambahan
                    if 'trades' in res and len(res['trades']) > 0:
                        st.caption(f"Sample trade terakhir: Entry {res['trades'][-1].get('entry_price', 'N/A')} → {res['trades'][-1].get('result', 'N/A')}")
                        
                elif isinstance(res, dict) and 'error' in res:
                    st.warning(f"⚠️ {res['error']}")
                else:
                    st.info("Data backtesting tersedia")
                    
            except Exception as e:
                st.error(f"Backtesting error: {str(e)}")
            
        with cols[1]:
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
                    analyzer.discord.send_watchlist_added(ticker) 
                    st.success(f"{ticker} added to watchlist")
                else:
                    st.info(f"{ticker} is already in watchlist")


        st.markdown("---")
        st.markdown("🎯 **Prediksi 5 Hari Mendatang**")

        # Button untuk trigger prediksi
        if st.button(f"🔮 Tampilkan Prediksi Detail", key=f"pred_detail_{ticker}"):
            with st.spinner(f"Menghitung prediksi untuk {ticker}..."):
                pred = analyzer.predict_price_direction(ticker, forecast_days=5)
            
            if "error" in pred:
                st.error(f"Prediksi gagal: {pred['error']}")
            else:
                # Tampilkan dalam format bagus
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                
                with col_pred1:
                    st.metric(
                        "Hari Naik", 
                        f"{pred['up_days']} hari", 
                        f"{pred['up_probability']*100:.0f}% prob"
                    )
                
                with col_pred2:
                    st.metric(
                        "Hari Turun", 
                        f"{pred['down_days']} hari", 
                        f"{pred['down_probability']*100:.0f}% prob"
                    )
                
                with col_pred3:
                    direction_emoji = "📈" if pred['net_direction'] == "UP" else "📉" if pred['net_direction'] == "DOWN" else "↔️"
                    st.metric(
                        "Arah Net", 
                        f"{direction_emoji} {pred['net_direction']}",
                        f"{pred['confidence']*100:.0f}% confidence"
                    )
                
                # Persentase pergerakan
                st.markdown("**📊 Estimasi Persentase:**")
                col_pct1, col_pct2 = st.columns(2)
                
                with col_pct1:
                    st.info(f"**Rata2 Naik:** {pred['expected_up_pct']}% per hari")
                
                with col_pct2:
                    st.info(f"**Rata2 Turun:** {pred['expected_down_pct']}% per hari")
                
                # Interpretasi
                if pred['net_direction'] == "UP":
                    st.success(f"**AI Prediksi:** {ticker} cenderung naik {pred['up_days']} dari 5 hari ke depan")
                elif pred['net_direction'] == "DOWN":
                    st.warning(f"**AI Prediksi:** {ticker} cenderung turun {pred['down_days']} dari 5 hari ke depan")
                else:
                    st.info(f"**AI Prediksi:** {ticker} cenderung sideway")    

        st.markdown("---")
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

## ============================================================
# 🤖 AI Kesimpulan Rekomendasi Jangka Pendek (1–3 Hari)
# ============================================================
if ranked:
    st.markdown("### 🤖 Kesimpulan AI Rekomendasi Jangka Pendek (1–3 Hari)")

    top5 = ranked[:5]
    composite_results = []
    macro = res_pack.get("macro", {})
    market_sentiment = macro.get("market_sentiment", "Neutral")

    # =======================
    # SCORING PER SAHAM
    # =======================
    for item in top5:
        # HANDLE BERBAGAI STRUKTUR DATA
        if isinstance(item, dict):
            ticker = item.get("ticker")
            score = item.get("score", 0)
        elif isinstance(item, (list, tuple)):
            ticker = item[0] if len(item) > 0 else None
            score = item[1] if len(item) > 1 else 0
        elif isinstance(item, str):
            ticker = item
            score = 0
        else:
            continue
        
        if not ticker:
            continue

        analysis = res_pack["results"].get(ticker)
        if not analysis:
            continue

        df = analysis["price_data"]
        last = df.iloc[-1]
        fm = analysis["fundamental_metrics"]
        plan = analysis["3day_plan"]
        ts = analysis["technical_score"]

        # ----- TECHNICAL INDICATORS -----
        rsi = 100 - abs(50 - last.get("RSI_14", 50)) * 2
        macd = np.clip(50 + last.get("MACD_Hist", 0) * 10, 0, 100)
        ma_trend = ts["components"].get("ma_trend_score", 50)
        vol_ratio = min(last.get("Vol_Ratio", 1) * 50, 100)
        ti_score = np.mean([rsi, macd, ma_trend, vol_ratio])

        # ----- DEEP TECHNICAL -----
        adx = last.get("ADX_14", 25)
        ema_cross = 100 if last.get("EMA_8", 0) > last.get("EMA_21", 0) > last.get("EMA_50", 0) else 50
        obv_trend = 70 if last.get("OBV", 0) > df["OBV"].mean() else 40
        vwap = 70 if last.get("Close", 0) > last.get("VWAP", 0) else 40
        deep_score = np.mean([adx, ema_cross, obv_trend, vwap])

        # ----- FUNDAMENTAL METRICS -----
        roe = min(fm.get("roe", 0), 30)
        der = 100 - min(fm.get("der", 1) * 10, 100)
        pe = 100 - min(fm.get("pe_ratio", 15) * 2, 100)
        pb = 100 - min(fm.get("pb_ratio", 1.5) * 30, 100)
        fm_score = np.mean([roe, der, pe, pb])

        # ----- MACRO CONTEXT -----
        macro_adj = 0
        if market_sentiment == "Risk-on": macro_adj = 10
        elif market_sentiment == "Risk-off": macro_adj = -10

        recomm_score = ts["score"]

        # Final composite weighted score
        final = (
            0.25 * ti_score +
            0.20 * deep_score +
            0.20 * fm_score +
            0.10 * (50 + macro_adj) +
            0.25 * recomm_score
        )

        composite_results.append({
            "ticker": ticker,
            "final_score": round(final, 2),
            "ti_score": ti_score,
            "deep_score": deep_score,
            "fm_score": fm_score,
            "macro_adj": macro_adj,
            "recommend_score": recomm_score,
            "signal": ts["signal"],
            "trend": "Uptrend" if ma_trend > 60 else "Sideway" if 40 <= ma_trend <= 60 else "Downtrend",
            "plan": plan,
        })

    # ============================================================
    # ROBUST FILTER → Paling penting!
    # ============================================================
    robust_filtered = []

    for r in composite_results:

        # Composite score minimal
        if r["final_score"] < 70:
            continue

        # Trend harus cocok dengan sinyal
        if r["signal"] == "BUY" and r["trend"] != "Uptrend":
            continue
        if r["signal"] == "SELL" and r["trend"] != "Downtrend":
            continue

        # Hindari noise RSI/ti score ekstrem
        if not (35 <= r["ti_score"] <= 85):
            continue

        # Entry–Stop–Target harus valid
        entry = r["plan"]["entry"]
        tgt   = r["plan"]["target"]
        sl    = r["plan"]["stop"]
        rr = abs(tgt - entry) / max(abs(entry - sl), 1e-6)

        if rr < 1:   # minimal R:R 1:1
            continue

        robust_filtered.append(r)

    # ============================================================
    # Jika tidak ada saham layak → tampilkan warning, stop!
    # ============================================================
    if not robust_filtered:
        st.markdown("""
        ### ⚠️ Tidak Ada Rekomendasi Layak Hari Ini
        AI tidak menemukan saham dengan probabilitas yang cukup kuat dan robust
        untuk trading jangka pendek (1–3 hari).

        Biasanya disebabkan:
        - Trend tidak valid
        - Volume rendah
        - Sinyal berlawanan arah trend
        - Market sedang risk-off / bearish
        """)
        st.stop()

    # =======================
    # Market Strength
    # =======================
    avg_market_strength = sum(r["final_score"] for r in robust_filtered) / len(robust_filtered)

    if avg_market_strength >= 80:
        market_view = "📈 Momentum pasar sangat kuat — peluang kenaikan tinggi dalam 1–3 hari."
        color = "#16a34a"
    elif avg_market_strength >= 60:
        market_view = "⚖️ Pasar konsolidasi dengan bias positif."
        color = "#facc15"
    else:
        market_view = "📉 Tekanan jual tinggi — potensi koreksi jangka pendek."
        color = "#dc2626"

    best_pick = max(robust_filtered, key=lambda x: x["final_score"])

    # =======================
    # RENDER BOX HTML
    # =======================
    html_kete = f"""
    <div style='background:linear-gradient(90deg,#0F172A,#1E293B);border:1px solid #334155;border-radius:14px;padding:1.2rem;margin-bottom:1rem;'>

    <h4 style='color:{color};margin-top:0'>{market_view}</h4>

    <p style='color:#CBD5E1'>
        📊 <b>Rata-rata kekuatan pasar:</b> {avg_market_strength:.2f}/100<br>
        🌍 <b>Sentimen makro:</b> {market_sentiment}<br>
        🔎 <b>Saham dianalisa:</b> {len(robust_filtered)} yang lolos robust filter
    </p>

    <div style='background-color:#0F172A;padding:1rem;margin-top:1rem;border-radius:12px;border:1px solid {color};'>
    <h4 style='color:{color};margin-bottom:0.4rem;'>🔥 Saham Paling Kuat untuk Trading Harian</h4>
    <p style='font-size:1.05rem;margin:0;'>
        <b>{best_pick['ticker']}</b> — {best_pick['signal']} ({best_pick['trend']})<br>
        Entry: <code>{best_pick['plan']['entry']}</code> | 🎯 Target: <code>{best_pick['plan']['target']}</code> | 🛑 Stop: <code>{best_pick['plan']['stop']}</code><br>
        Composite Score: <b>{best_pick['final_score']:.2f}</b> / 100
    </p>
    </div>

    <p style='color:#94a3b8;margin-top:1rem;font-size:0.9rem;'>
    ⚠️ Rekomendasi hanya muncul jika sinyal betul-betul robust berdasarkan Technical + Deep Technical + Fundamental + Macro + Trend.
    </p>

    </div>
    """

    # ============================================================
    # 🔮 AI Prediksi 1–3 Hari (Hybrid Model)
    # ============================================================
    st.markdown("""
    <br>
    <h2 style='margin-bottom:0.2rem;'>🔮 Prediksi AI 1–3 Hari (Hybrid Model)</h2>
    <p style='color:#94a3b8;margin-top:-0.4rem;'>ARIMA + ML + ATR + Sentimen Makro</p>
    """, unsafe_allow_html=True)

    selected_ticker = st.session_state.get("selected_ticker", None)

    if selected_ticker:
        with st.spinner(f"🔄 Menghitung prediksi hybrid untuk {selected_ticker} ..."):
            pred = analyzer.predict_3days(selected_ticker)

        if "error" in pred:
            st.error(f"Hybrid predictor gagal: {pred['error']}")
        else:

            # ---- CARD WRAPPER ----
            st.markdown("""
            <div style='background:#0f172a;padding:1.5rem;border-radius:14px;border:1px solid #1e293b;margin-top:1rem;'>
            """, unsafe_allow_html=True)

            # ---- SIGNAL BESAR ----
            sig_color = {
                "BUY": "#22c55e",
                "SELL": "#ef4444",
                "HOLD": "#facc15"
            }.get(pred["final_signal"], "#f1f5f9")

            st.markdown(
                f"""
                <h3 style='color:{sig_color};text-align:center;margin-top:0;font-size:1.9rem;'>
                    {pred['final_signal']}
                </h3>
                <p style='color:#94a3b8;text-align:center;margin-top:-0.6rem;'>
                    Macro Bias: <b>{pred['macro_bias']}</b>
                </p>
                """,
                unsafe_allow_html=True
            )

            # ---- METRICS 3 KOLOM ----
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Return (3D)", f"{pred['expected_return_pct']} %")
            c2.metric("Confidence", f"{pred['confidence'] * 100:.1f} %")
            c3.metric("Volatility (ATR)", pred["volatility_index"])

            st.markdown("<br>", unsafe_allow_html=True)

            # ---- CHART PLOTLY ----
            import plotly.graph_objects as go

            future_x = ["+1 Hari", "+2 Hari", "+3 Hari"]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=future_x,
                y=pred["predicted_prices"],
                mode="lines+markers",
                name="Prediksi",
                line=dict(width=3)
            ))
            fig.add_trace(go.Scatter(
                x=future_x,
                y=pred["ci_upper"],
                mode="lines",
                name="Confidence Upper",
                line=dict(dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=future_x,
                y=pred["ci_lower"],
                mode="lines",
                name="Confidence Lower",
                line=dict(dash="dash")
            ))

            fig.update_layout(
                title=f"Prediksi Harga {selected_ticker} (3 Hari ke Depan)",
                yaxis_title="Harga",
                xaxis_title="Hari Prediksi",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # ---- TUTUP CARD WRAPPER ----
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Pilih ticker terlebih dahulu untuk mendapatkan prediksi hybrid 1–3 hari.")
        
    render_html(html_kete)



# ---------------------------
# Watchlist panel
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")
wl = st.session_state.get("watchlist", [])
st.sidebar.write("Saved tickers:")
for t in wl:
    st.sidebar.write(f"- {t}")
# if st.sidebar.button("Export Watchlist CSV"):
#     if wl:
#         wl_df = pd.DataFrame(wl, columns=["Ticker"])
#         csv_bytes = wl_df.to_csv(index=False).encode("utf-8")
#         st.sidebar.download_button("Download watchlist", data=csv_bytes, file_name="watchlist.csv", mime="text/csv")
#     else:
#         st.sidebar.info("Watchlist kosong.")

# if st.sidebar.button(f"📢 Kirim sinyal ke Discord"):
#     result = analysis.get(ticker, {})
#     signal = result.get("signal", "")
#     if "BUY" in signal or "SELL" in signal or "STRONG BUY" in signal or "STRONG SELL" in signal:
#         price = float(result.get("last_price", 0))
#         conf = result.get("score", 0) / 100
#         summary = result.get("timing_advice", "")
#         analyzer.discord.send_message(ticker, signal, price, conf, summary)
#         st.success(f"Notifikasi {signal} {ticker} dikirim ke Discord!")
#     else:
#         st.warning(f"Tidak ada sinyal BUY/SELL untuk {ticker} ({signal}).")


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
