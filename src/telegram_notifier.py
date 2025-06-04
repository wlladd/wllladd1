# src/telegram_notifier.py

import requests
import logging

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}/sendMessage"

    def send_message(self, text: str):
        if not self.token or not self.chat_id:
            return
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(self.base_url, data=payload)
            if response.status_code != 200:
                logging.error(f"Ошибка отправки Telegram-сообщения: {response.text}")
        except Exception as e:
            logging.error(f"Исключение при отправке Telegram: {e}", exc_info=True)

    def send_order_opened(self, side: str, symbol: str, price: float, volume: float, reason: str):
        if not self.token or not self.chat_id:
            return
        msg = f"♟ <b>Ордер открыт:</b>\n" \
              f"Тип: {side}\n" \
              f"Инструмент: {symbol}\n" \
              f"Цена: {price:.5f}\n" \
              f"Объем: {volume:.3f} LOT\n" \
              f"Причина: {reason}"
        self.send_message(msg)

    def send_order_closed(self, side: str, symbol: str, price_entry: float, price_exit: float, volume: float, profit: float, reason: str):
        if not self.token or not self.chat_id:
            return
        pnl = f"+{profit:.2f}" if profit >= 0 else f"{profit:.2f}"
        msg = f"🛑 <b>Ордер закрыт:</b>\n" \
              f"Тип: {side}\n" \
              f"Инструмент: {symbol}\n" \
              f"Entry: {price_entry:.5f}\n" \
              f"Exit:  {price_exit:.5f}\n" \
              f"Объем: {volume:.3f} LOT\n" \
              f"P/L: {pnl} USD\n" \
              f"Причина: {reason}"
        self.send_message(msg)

    def send_error(self, error_text: str):
        if not self.token or not self.chat_id:
            return
        msg = f"❗ <b>Ошибка:</b>\n<pre>{error_text}</pre>"
        self.send_message(msg)
