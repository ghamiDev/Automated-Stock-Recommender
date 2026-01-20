import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ======================================================
# CONFIG
# ======================================================
API_KEY = "tptOmOGppwPnnAao9iNqzD5ZmFIH4RD0"
BASE_URL = "https://api.massive.com/v3"
SYMBOL = "BBRI.JK"  # ganti sesuai kebutuhan: BBRI.JK, BBCA.JK dll

# https://api.massive.com/v3/snapshot?ticker=NVDA&order=asc&limit=10&sort=ticker&apiKey=tptOmOGppwPnnAao9iNqzD5ZmFIH4RD0
# ======================================================
# FETCH REAL-TIME PRICE
# ======================================================
def get_price(symbol: str):

    try:
        url = f"{BASE_URL}/snapshot?ticker={symbol}&order=asc&limit=10&sort=ticker"
        headers = {"Authorization": f"Bearer {API_KEY}"}

        r = requests.get(url, headers=headers)
        data = r.json()

        print(data)
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["last"]

        return None

    except Exception as e:
        return None



# ======================================================
# SIMPLE MOVING AVERAGE
# ======================================================
def moving_average(values, period=5):
    if len(values) < period:
        return None
    return np.mean(values[-period:])


# ======================================================
# STREAMLIT UI
# ======================================================
st.title("📈 Day Trading Bot – MA Strategy + TP 2% + SL 1%")
st.write("Menggunakan Massive API + Streamlit")

symbol = st.text_input("Symbol saham", SYMBOL)

run_bot = st.button("▶ Start Trading Bot")

st.write("---")

log_box = st.empty()
price_chart = st.line_chart()

# ======================================================
# TRADING LOOP
# ======================================================
if run_bot:
    st.success(f"Bot berjalan untuk {symbol} ...")
    
    is_holding = False
    buy_price = 0
    price_history = []

    while True:
        price = get_price(symbol)
       
        if price is None:
            log_box.write("❌ Gagal mengambil data harga.")
            time.sleep(5)
            continue

        price_history.append(price)
        ma5 = moving_average(price_history, 5)

        # Update grafik
        price_chart.add_rows({"price": price})

        log_text = f"Harga sekarang: {price} | MA5: {ma5}"
        log_box.write(log_text)

        # ======================================================
        # ENTRY RULE: BUY
        # ======================================================
        if not is_holding and ma5 and price > ma5:
            is_holding = True
            buy_price = price
            log_box.write(f"🚀 BUY pada harga {buy_price}")

        # ======================================================
        # EXIT RULE: SELL
        # ======================================================
        if is_holding:
            TP = buy_price * 1.02
            SL = buy_price * 0.99

            if price >= TP:
                log_box.write(f"🎯 TAKE PROFIT! SELL di {price}")
                is_holding = False

            elif price <= SL:
                log_box.write(f"🛑 STOP LOSS! SELL di {price}")
                is_holding = False

            elif ma5 and price < ma5:
                log_box.write(f"⚠ Exit MA cross-down. SELL di {price}")
                is_holding = False

        time.sleep(5)  # update setiap 5 detik
