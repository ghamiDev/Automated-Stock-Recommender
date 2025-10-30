import os
import requests
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger("DiscordNotifier")
logging.basicConfig(level=logging.INFO)


class DiscordNotifier:
    """
    DiscordNotifier dengan dukungan:
    - Pengiriman pesan teks & embed
    - Cooldown per ticker+signal (anti spam)
    - Thread-safe
    """

    def __init__(self, webhook_url: Optional[str] = None, cooldown: int = 3600):
        """
        :param webhook_url: URL webhook Discord (bisa None untuk disable)
        :param cooldown: waktu minimal antar notifikasi sama (detik)
        """
        # Ambil dari argumen atau .env
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK")
        self.cooldown = cooldown
        self._lock = threading.Lock()
        self._last_sent = {}  # (ticker, signal): timestamp terakhir

        if not self.webhook_url:
            logger.warning("⚠️ Webhook URL belum diatur, notifikasi tidak akan dikirim.")

    # ------------------------------------------------------------------
    # Helper internal untuk anti-spam
    # ------------------------------------------------------------------
    def _can_send(self, ticker: str, signal: str) -> bool:
        """Cek apakah notifikasi boleh dikirim (berdasarkan cooldown)."""
        key = (ticker, signal)
        now = time.time()
        with self._lock:
            last_time = self._last_sent.get(key, 0)
            if now - last_time >= self.cooldown:
                self._last_sent[key] = now
                return True
            return False

    # ------------------------------------------------------------------
    # Kirim pesan teks biasa
    # ------------------------------------------------------------------
    def send_message(self, content: str, ticker: Optional[str] = None, signal: Optional[str] = None):
        if not self.webhook_url or not content:
            return

        # Filter spam via cooldown
        if ticker and signal and not self._can_send(ticker, signal):
            logger.info(f"⏸ Skip notif teks (cooldown aktif) {ticker}:{signal}")
            return

        payload = {"content": content}
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=8)
            if response.status_code != 204:
                logger.warning(f"⚠️ Discord API response: {response.status_code} {response.text}")
            else:
                logger.info(f"✅ Notifikasi teks terkirim ke Discord: {content[:60]}...")
        except Exception as e:
            logger.error(f"[DiscordNotifier] Gagal kirim pesan teks: {e}")

    # ------------------------------------------------------------------
    # Kirim pesan embed berwarna (lebih informatif)
    # ------------------------------------------------------------------
    def send_embed(self, title: str, description: str, color: int = 0x808080,
                   ticker: Optional[str] = None, signal: Optional[str] = None):
        """
        Kirim notifikasi embed ke Discord.
        Warna: hex integer (contoh 0x16A34A = hijau).
        """
        if not self.webhook_url or not title:
            return

        # Filter spam via cooldown
        if ticker and signal and not self._can_send(ticker, signal):
            logger.info(f"⏸ Skip notif embed (cooldown aktif) {ticker}:{signal}")
            return

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "footer": {"text": "📈 Automated Stock Recommender"},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        payload = {"embeds": [embed]}
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=8)
            if response.status_code not in (200, 204):
                logger.warning(f"⚠️ Discord API embed response: {response.status_code} {response.text}")
            else:
                logger.info(f"✅ Embed terkirim ke Discord: {title}")
        except Exception as e:
            logger.error(f"[DiscordNotifier] Gagal kirim embed: {e}")

    
    # ------------------------------------------------------------------
    # Kirim notifikasi ketika saham ditambahkan ke watchlist
    # ------------------------------------------------------------------
    def send_watchlist_added(self, ticker: str):
        """
        Kirim pesan embed sederhana saat user menambahkan saham ke watchlist.
        Tidak terkena cooldown (boleh langsung kirim).
        """
        if not self.webhook_url or not ticker:
            return

        embed = {
            "title": f"⭐ Ditambahkan ke Watchlist",
            "description": f"Saham **{ticker}** telah ditambahkan ke watchlist dan akan dimonitor secara real-time.",
            "color": 0x3B82F6,  # biru
            "footer": {"text": "📈 Automated Stock Recommender"},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        payload = {"embeds": [embed]}
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=8)
            if response.status_code not in (200, 204):
                logger.warning(f"⚠️ Gagal kirim notifikasi watchlist: {response.status_code} {response.text}")
            else:
                logger.info(f"✅ Notifikasi watchlist terkirim untuk {ticker}")
        except Exception as e:
            logger.error(f"[DiscordNotifier] Gagal kirim notifikasi watchlist: {e}")

