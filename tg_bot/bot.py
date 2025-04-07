import json
import os
import tempfile
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from predict import predict_image

# Конфигурация
BOT_TOKEN = '7902173052:AAHTzM91bNYninOEKFVjyktVZhgz9BV8DKQ'
LABELS_PATH = 'labels.json'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    context.user_data['stopped'] = False  # Сбрасываем флаг остановки
    await update.message.reply_text('Привет! Отправь мне фото достопримечательности для распознавания.')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /stop"""
    context.user_data['stopped'] = True  # Устанавливаем флаг остановки
    user = update.effective_user
    await update.message.reply_text(
        f"Работа бота приостановлена, {user.first_name}!\n"
        "Чтобы снова активировать бота, отправьте /start"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик фотографий"""
    temp_path = None
    try:
        if context.user_data.get('stopped', False):
            await update.message.reply_text("Бот приостановлен. Для активации отправьте /start")
            return

        # Скачивание фото
        photo_file = await update.message.photo[-1].get_file()
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            temp_path = f.name
            await photo_file.download_to_drive(temp_path)

        # Получаем предсказание
        prediction = str(predict_image(temp_path))
        
        # Загружаем метки
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        # Формируем ответ
        landmark_info = labels.get(prediction, {})
        response = (
            "🌍 *Информация о достопримечательности:*\n"
            f"🏛 *Название:* {landmark_info.get('name', 'Неизвестно')}\n"
            f"📜 *Описание:* {landmark_info.get('info', 'Нет описания')}\n"
            f"📍 *Местоположение:* {landmark_info.get('location', 'Адрес не указан')}\n"
            f"💡 *Советы туристам:* {landmark_info.get('tips', 'Рекомендации отсутствуют')}"
        )

        await update.message.reply_markdown(response)

    except Exception as e:
        await update.message.reply_text(f'⛔ Ошибка: {str(e)}')
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    """Запуск бота"""
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    application.run_polling()

if __name__ == '__main__':
    main()