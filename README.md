# 📈 **Automated Stock Recommender Dashboard**

Aplikasi ini menganalisis saham secara otomatis dan menampilkan **rekomendasi trading harian** berdasarkan kombinasi **analisis teknikal**, **fundamental**, dan **makroekonomi**.  
Seluruh proses berjalan **otomatis dan real-time** menggunakan data dari **Yahoo Finance (yfinance)**.

---

## 🚀 **Fitur Utama**

### 🔹 1. Dashboard Rekomendasi Otomatis

- Menampilkan **saham-saham terbaik** untuk ditradingkan hari ini.
- Setiap saham menampilkan:
  - Nama ticker dan nama perusahaan  
  - Harga terakhir & perubahan persen  
  - Sinyal: `BUY`, `HOLD`, atau `SELL`  
  - Strength Score (0–100)  
  - Ikon tren (⬆️ Uptrend, ↔️ Sideway, ⬇️ Downtrend)
- Warna dinamis:
  - 🟢 Hijau = sinyal beli  
  - 🟡 Kuning = hold  
  - 🔴 Merah = jual  

### 🔹 2. Analisis Detail (Expandable)

Klik “Detail” untuk membuka analisis lengkap:
- Mini **grafik candlestick interaktif** dengan overlay MA/EMA  
- **Indikator teknikal lengkap:** RSI, MACD, MA, ATR, Volume Ratio  
- **Analisa fundamental:** PER, PBV, ROE, DER, Dividend Yield  
- **3-Day Trading Plan:** Entry, Stop Loss, Target, Position Size  
- **AI Summary** otomatis untuk ringkasan rekomendasi singkat  

### 🔹 3. Watchlist Pribadi

- Tambahkan saham ke watchlist dengan satu klik  
- Watchlist tersimpan di session & bisa diunduh ke CSV  

### 🔹 4. Analisis Makroekonomi

- Sidebar menampilkan indikator makro utama (IDX, kurs, sentimen pasar)  
- Disertakan konteks “market sentiment” secara otomatis  

---

## ⚙️ **Struktur File**

```
📁 project/
│
├── analyzer_full.py     ← Modul analisis saham (fundamental, teknikal, makro)
├── gui_full.py          ← Aplikasi utama Streamlit (dashboard)
├── requirements.txt     ← Daftar dependency
└── README.md            ← File dokumentasi ini
```

---

## 💾 **Instalasi**

1. **Buat virtual environment (opsional tapi disarankan):**
   ```bash
   python -m venv saham_env
   source saham_env/bin/activate   # (Linux/Mac)
   saham_env\Scripts\activate      # (Windows)
   ```

2. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**
   ```bash
   streamlit run gui_full.py
   ```

4. Buka browser dan akses:  
   👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 **Library yang Digunakan**

| Library               | Kegunaan                             |
| --------------------- | ------------------------------------ |
| **streamlit**         | UI dashboard interaktif              |
| **yfinance**          | Ambil data saham & fundamental       |
| **pandas**, **numpy** | Manipulasi & analisis data           |
| **plotly**            | Grafik candlestick & indikator       |
| **typing-extensions** | Dukungan type hint untuk Python 3.8+ |

---

## 💼 **Analisis yang Disertakan**

| Jenis Analisis         | Komponen                                               |
| ---------------------- | ------------------------------------------------------ |
| **Teknikal**           | MA, EMA, RSI, MACD, ATR, Stochastic, Volume Ratio      |
| **Fundamental**        | EPS, PER, PBV, ROE, DER, Profit Margin, Dividend Yield |
| **Makroekonomi**       | Indeks pasar, Sentimen, Sektor dominan                 |
| **AI Decision System** | Menghitung composite score (0–100) untuk rekomendasi   |

---

## 📋 **Kategori Sinyal**

| Skor   | Kategori        | Warna     | Rekomendasi          |
| ------ | --------------- | --------- | -------------------- |
| 80–100 | **STRONG BUY**  | 🟢 Hijau  | Momentum sangat kuat |
| 60–79  | **BUY**         | 🟡 Kuning | Potensi naik         |
| 40–59  | **HOLD**        | ⚪ Abu    | Netral               |
| 20–39  | **SELL**        | 🟠 Oranye | Potensi turun        |
| 0–19   | **STRONG SELL** | 🔴 Merah  | Tren turun kuat      |

---

## 🧠 **AI Summary (Contoh Keluaran)**

> **BBCA.JK** menunjukkan tren naik kuat dengan sinyal beli jangka pendek.  
> RSI 58 masih netral dan harga berpotensi menuju 9,500.  
> Entry disarankan di 9,350, Stop Loss 9,200, Target 9,600.  
> Skor keseluruhan: 87/100.

---

## 🛠️ **Kustomisasi**

- Edit daftar ticker default di sidebar.  
- Ubah periode data (`1mo`, `3mo`, `6mo`, `1y`).  
- Atur `Top N Recommendations`, `Capital`, dan `Risk per Trade`.  
- Pilihan auto-refresh tiap X detik.  

---

## ⚠️ **Disclaimer**

> Aplikasi ini dibuat untuk **tujuan edukasi dan analisis**.  
> Data diambil dari Yahoo Finance dan **tidak menjamin real-time akurat**.  
> Gunakan hasil analisa sebagai **alat bantu keputusan**, bukan rekomendasi investasi resmi.  

---

## 🧾 **Lisensi**

MIT License © 2025
