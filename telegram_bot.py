import os
import time
import asyncio
from telegram import Bot
from telegram.error import TelegramError

TELEGRAM_TOKEN = '7033304819:AAFkQ27U6M9Cxj-dsePyjev1AAjsyF9biTU'
CHAT_ID = '933193998'
bot = Bot(token=TELEGRAM_TOKEN)


last_notification_time = 0
notification_interval = 10

async def send_telegram_notification(detection):
    global last_notification_time
    current_time = time.time()

    if current_time - last_notification_time >= notification_interval:
        message = (f"Выявлено нарушение!\n"
                   f"Нарушение: {detection['label']}\n"
                   f"Время: {detection['time']}\n"
                   f"Объект: {detection['video']}")
        snapshot_path = os.path.join('static/detections/', detection['snapshot'])
        try:
            await bot.send_photo(chat_id=CHAT_ID, photo=open(snapshot_path, 'rb'), caption=message)
            last_notification_time = current_time  #
        except TelegramError as e:
            print(f"Failed to send Telegram notification: {e}")
