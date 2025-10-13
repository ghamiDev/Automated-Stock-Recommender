import pandas as pd
import streamlit as st
from pathlib import Path

EXCEL_FILE = Path("Daftar_Saham.xlsx")


@st.cache_data(show_spinner=False)
def get_idx_tickers():
   
    if not EXCEL_FILE.exists():
        st.error(f"❌ File tidak ditemukan: {EXCEL_FILE}")
        return []

    try:
        df = pd.read_excel(EXCEL_FILE)

        # Cari kolom yang berisi kode emiten
        possible_cols = [c for c in df.columns if "Code" in c or "Kode" in c or "Emiten" in c]
        if not possible_cols:
            st.error("⚠️ Tidak ditemukan kolom kode emiten di file Excel.")
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
        tickers = sorted(list(set(tickers)))

        return tickers

    except Exception as e:
        st.error(f"❌ Gagal membaca file Excel: {e}")
        return []
