import logging
import os
import re
import asyncio
import time
from dotenv import load_dotenv
import nest_asyncio

import openai
import faiss
import numpy as np

# Для конвертации аудио форматов
from pydub import AudioSegment

# Импорт функции получения эмбеддингов из нового API OpenAI
from openai.embeddings_utils import get_embedding

# Импорт локальной модели Whisper для транскрибации
import whisper

# Загружаем модель Whisper "large" (либо "Whisper Real Time", если доступно)
WHISPER_MODEL = whisper.load_model("large")
logger = logging.getLogger(__name__)
logger.info("Whisper модель 'large' загружена.")

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Применяем nest_asyncio для корректной работы вложенных циклов событий
nest_asyncio.apply()

# Загружаем переменные окружения из файла .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))
openai.api_key = OPENAI_API_KEY

# Проверяем переменные окружения
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Переменная TELEGRAM_BOT_TOKEN не задана в файле .env")
if not OPENAI_API_KEY:
    raise ValueError("Переменная OPENAI_API_KEY не задана в файле .env")

# Настройка логирования: вывод в консоль и в файл user_requests.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("user_requests.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# URL-ы фотографий руководителей
PHOTO_PRESIDENT = "https://static.tildacdn.com/tild6139-3564-4162-a636-613162376438/ca38d0e45e485eeaa214.jpg"
PHOTO_VP_URL = "https://static.tildacdn.com/tild3033-3264-4032-a362-393764633238/12312312314444444444.jpg"

# Преимущества членства
MEMBERSHIP_ADVANTAGES = (
    "• Развитие профессиональных навыков: участие в образовательных программах, мастер-классах и вебинарах.\n"
    "• Расширение деловых связей: обмен опытом с единомышленниками и поиск партнёров.\n"
    "• Юридическая и административная поддержка: бесплатные консультации юристов и доступ к шаблонам документов.\n"
    "• Индивидуальные консультации: личный разбор бизнес-задач.\n"
    "• Экономические выгоды: скидки и эксклюзивные предложения от партнёров."
)

# Биография президента (пример текста)
BIOGRAPHY_PRESIDENT = """
**Калинин Александр Сергеевич** возглавляет Опору России на федеральном уровне. Он осуществляет стратегическое руководство организацией и является её президентом.

**Биография:**
- Родился: 4 ноября в г. Челябинске.
- Образование: окончил Челябинский политехнический институт, обучался в аспирантуре, окончил Уральскую академию государственной службы.
- Деятельность: основал предприятие в торговле, создал производственную компанию, с 2002 г. участвует в деятельности «Опоры России», с 2014 г. — президент.
"""

# --- Функции для работы с базой знаний ---

def convert_docx_to_txt(input_filename: str = "russia_base.docx", output_filename: str = "russia_base.txt") -> bool:
    try:
        from docx import Document
    except ImportError:
        logger.error("Не установлена библиотека python-docx. Установите её: pip install python-docx")
        return False
    try:
        document = Document(input_filename)
        full_text = [para.text for para in document.paragraphs]
        text = "\n".join(full_text)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Файл {input_filename} успешно конвертирован в {output_filename}.")
        return True
    except Exception as e:
        logger.error(f"Ошибка при конвертации файла {input_filename}: {e}")
        return False

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < text_length else text_length
    return chunks

async def get_embedding_for_chunk(chunk: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    retries = 3
    for attempt in range(retries):
        try:
            vector = await loop.run_in_executor(
                None, lambda: get_embedding(chunk, engine="text-embedding-ada-002")
            )
            return np.array(vector, dtype="float32")
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга для чанка (попытка {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                return np.zeros(1536, dtype="float32")
    return np.zeros(1536, dtype="float32")

async def load_and_index_base_async(filename: str = "russia_base.txt", docx_filename: str = "russia_base.docx", chunk_size: int = 500, overlap: int = 50):
    if not os.path.exists(filename):
        if os.path.exists(docx_filename):
            success = convert_docx_to_txt(docx_filename, filename)
            if not success:
                logger.error("Не удалось получить текстовую версию базы знаний.")
                return faiss.IndexFlatL2(1536), []
        else:
            logger.warning(f"Файл базы знаний {docx_filename} не найден. Продолжаем без базы знаний.")
            return faiss.IndexFlatL2(1536), []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Ошибка чтения файла базы знаний: {e}")
        return faiss.IndexFlatL2(1536), []
    
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    embeddings = await asyncio.gather(*(get_embedding_for_chunk(chunk) for chunk in chunks))
    embeddings = np.stack(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"Создан FAISS индекс с {index.ntotal} элементами")
    return index, chunks

global_index = None
global_chunks = []

# --- Функция для поиска контекста в базе знаний ---
async def retrieve_context(query: str, k: int = 3) -> str:
    if global_index is None or not global_chunks:
        return ""
    loop = asyncio.get_running_loop()
    query_embedding = await loop.run_in_executor(
        None, lambda: np.array(get_embedding(query, engine="text-embedding-ada-002"), dtype="float32")
    )
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = global_index.search(query_embedding, k)
    context_chunks = [global_chunks[i] for i in indices[0] if i < len(global_chunks)]
    logger.info("Использованные чанки из базы знаний:\n%s", "\n".join(context_chunks))
    return "\n".join(context_chunks)

# --- Функция для генерации ответа через GPT-4o-mini с добавлением контекста ---
async def generate_gpt_response(prompt: str) -> str:
    context_text = await retrieve_context(prompt, k=3)
    full_prompt = f"Используй следующую информацию из базы знаний:\n{context_text}\n\nОтветь на запрос: {prompt}"
    logger.info("Full prompt для GPT-4o-mini:\n%s", full_prompt)  # Вывод full_prompt для отладки
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Попробуйте временно переключиться на gpt-4 для проверки
            messages=[
                {"role": "system", "content": (
                    'Ты — Telegram-бот общественной организации "Опора России". Твоя задача — предоставлять информацию о руководстве "Опорой России", мероприятиях, услугах и преимуществах членства, основываясь на базе знаний. '
                    'Отвечай полно, информативно и в дружелюбном тоне.'
                    'Запоминай предыдущий ответ.'
                )},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
            max_tokens=3000,
        )
        text = response.choices[0].message["content"].strip()
        logger.info("Сгенерирован ответ от GPT-4: %s", text)
        return text
    except Exception as e:
        logger.error(f"Ошибка вызова OpenAI: {e}")
        return "Извините, произошла ошибка при формировании ответа."

# --- Обработчики команд и сообщений ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Пользователь {user.username} ({user.id}) ввёл команду /start.")
    welcome_text = (
        "Здравствуйте!\n\n"
        "Добро пожаловать в бот общественной организации \"Опора России\". "
        "Здесь вы можете узнать информацию о мероприятиях, услугах и преимуществах членства.\n\n"
        "Доступные команды:\n"
        "/start — запустить бота\n"
        "/help — справочная информация\n"
        "/stop — остановить бота (только для администратора)\n"
        "/getid — узнать свой Telegram ID\n\n"
        "Выберите интересующую опцию ниже:"
    )
    keyboard = [
        [InlineKeyboardButton("Актуальные мероприятия", callback_data="events")],
        [InlineKeyboardButton("Преимущества членства", callback_data="membership")],
        [InlineKeyboardButton("Показать руководителей", callback_data="show_photos")],
        [InlineKeyboardButton("Получить помощь", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def help_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Пользователь {user.username} ({user.id}) ввёл команду /help.")
    help_text = (
        "📚 **Помощь по боту «Опора России»** 📚\n\n"
        "**Команды:**\n"
        "/start — запустить бота\n"
        "/help — получить справочную информацию\n"
        "/stop — остановить бота (только для администратора)\n"
        "/getid — узнать свой Telegram ID\n\n"
        "**Примеры запросов:**\n"
        "- Кто сейчас возглавляет Опору России? Приведи справку с фото.\n"
        "- Кто руководит тюменским отделением?\n"
        "- Расскажи о преимуществах членства."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def stop_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user.id != ADMIN_USER_ID:
        await update.message.reply_text("У вас нет прав для выполнения этой команды.")
        logger.warning(f"Пользователь {user.username} ({user.id}) попытался использовать /stop без прав.")
        return
    logger.info(f"Пользователь {user.username} ({user.id}) ввёл /stop.")
    await update.message.reply_text("🔚 Разговор завершён. До свидания!")
    await context.application.stop()

# Обработчик для голосовых сообщений и аудиофайлов
async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Получено голосовое/аудио сообщение от пользователя {user.username} ({user.id}).")
    
    # Используем поле voice или, если его нет, audio
    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text("Не удалось распознать голосовое сообщение или аудиофайл.")
        return

    try:
        await update.message.reply_text("Ваше голосовое сообщение получено и обрабатывается...")
        logger.info("Отправлено подтверждение получения голосового сообщения.")

        file = await voice.get_file()
        timestamp = int(time.time())
        ogg_path = f"voice_{user.id}_{timestamp}.ogg"
        await file.download_to_drive(ogg_path)
        logger.info(f"Файл сохранен: {ogg_path} (размер: {os.path.getsize(ogg_path)} байт)")

        try:
            # Конвертируем OGG в WAV с помощью pydub
            audio_seg = AudioSegment.from_file(ogg_path, format="ogg")
            wav_path = ogg_path.replace(".ogg", ".wav")
            audio_seg.export(wav_path, format="wav")
            logger.info(f"Файл конвертирован в WAV: {wav_path}")
        except Exception as e:
            logger.error(f"Ошибка конвертации: {e}")
            await update.message.reply_text("⚠️ Не удалось конвертировать аудио. Проверьте установку ffmpeg!")
            os.remove(ogg_path)
            return

        transcription = ""
        try:
            # Транскрибация с использованием Whisper Large (язык "ru", fp16=False для CPU)
            result = WHISPER_MODEL.transcribe(wav_path, language="ru", fp16=False)
            transcription = result["text"].strip()
            transcription = re.sub(r'\bаппора\b', 'Опора', transcription, flags=re.IGNORECASE)
            logger.info(f"Распознанный текст: {transcription}")
        except Exception as e:
            logger.error(f"Ошибка транскрибации с Whisper Large: {e}")
            await update.message.reply_text("🔇 Не удалось распознать речь.")
            return

        try:
            os.remove(ogg_path)
            os.remove(wav_path)
            logger.info("Временные файлы удалены.")
        except Exception as e:
            logger.error(f"Ошибка удаления временных файлов: {e}")

        if transcription:
            await update.message.reply_text("💡 Распознанный текст: " + transcription)
            response = await generate_gpt_response(transcription)
            await update.message.reply_text(response)
            logger.info(f"Ответ отправлен пользователю {user.username} ({user.id}).")
        else:
            await update.message.reply_text("❌ Не удалось получить текст из голосового сообщения.")
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка при обработке голосового сообщения.")

# Обработчик для видео заметок
async def video_note_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Получена видео заметка от пользователя {user.username} ({user.id}).")
    try:
        await update.message.reply_text("Ваше голосовое сообщение (в формате видео) получено и обрабатывается...")
        video_note = update.message.video_note
        if not video_note:
            await update.message.reply_text("Не удалось получить видео заметку.")
            return

        file = await video_note.get_file()
        timestamp = int(time.time())
        mp4_path = f"video_note_{user.id}_{timestamp}.mp4"
        await file.download_to_drive(mp4_path)
        logger.info(f"Видео заметка скачана: {mp4_path}")

        try:
            # Извлекаем аудио из видео заметки с помощью pydub (ffmpeg должен быть установлен)
            audio_seg = AudioSegment.from_file(mp4_path, format="mp4")
            wav_path = mp4_path.replace(".mp4", ".wav")
            audio_seg.export(wav_path, format="wav")
            logger.info(f"Аудио извлечено и сохранено как WAV: {wav_path}")
        except Exception as e:
            logger.error(f"Ошибка извлечения аудио из видео заметки: {e}")
            await update.message.reply_text("⚠️ Не удалось извлечь аудио из видео заметки.")
            os.remove(mp4_path)
            return

        transcription = ""
        try:
            result = WHISPER_MODEL.transcribe(wav_path, language="ru", fp16=False)
            transcription = result["text"].strip()
            transcription = re.sub(r'\bаппора\b', 'Опора', transcription, flags=re.IGNORECASE)
            logger.info(f"Распознанный текст из видео заметки: {transcription}")
        except Exception as e:
            logger.error(f"Ошибка транскрибации видео заметки с Whisper Large: {e}")
            await update.message.reply_text("🔇 Не удалось распознать речь из видео заметки.")
            return

        try:
            os.remove(mp4_path)
            os.remove(wav_path)
            logger.info("Временные файлы видео заметки удалены.")
        except Exception as e:
            logger.error(f"Ошибка удаления временных файлов видео заметки: {e}")

        if transcription:
            await update.message.reply_text("💡 Распознанный текст: " + transcription)
            response = await generate_gpt_response(transcription)
            await update.message.reply_text(response)
            logger.info(f"Ответ отправлен пользователю {user.username} ({user.id}).")
        else:
            await update.message.reply_text("❌ Не удалось получить текст из видео заметки.")
    except Exception as e:
        logger.error(f"Ошибка при обработке видео заметки: {e}")
        await update.message.reply_text("⚠️ Произошла ошибка при обработке видео заметки.")

# Обработчик для текстовых сообщений
async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:
        user_text = update.message.text.strip()
        logger.info(f"Получено текстовое сообщение: {user_text}")
        if user_text.startswith("/"):
            return
        # Если в сообщении присутствует одно из ключевых слов
        keywords = ["опора", "преимущества", "вступления", "руководство"]
        if any(keyword in user_text.lower() for keyword in keywords):
            response = await generate_gpt_response(user_text)
            await update.message.reply_text(response)
            logger.info("Ответ GPT отправлен.")
        else:
            await update.message.reply_text("Ваш запрос не распознан. Попробуйте ввести другой вопрос или воспользуйтесь /help.")
    else:
        logger.info("Получено обновление без текстового сообщения.")


# Обработчик для inline-кнопок
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    user = update.effective_user
    logger.info(f"Пользователь {user.username} ({user.id}) нажал inline-кнопку: {data}")

    if data == "events":
        prompt = "Расскажи о предстоящих мероприятиях."
        response = await generate_gpt_response(prompt)
        await query.edit_message_text(text=response)
    elif data == "membership":
        prompt = "Расскажи о преимуществах членства в Опоре России."
        response = await generate_gpt_response(prompt)
        await query.edit_message_text(text=response)
    elif data == "show_photos":
        try:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=PHOTO_PRESIDENT,
                caption="**Александр Калинин** — Президент «Опора России»",
                parse_mode="Markdown"
            )
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=PHOTO_VP_URL,
                caption="**Эдуард Омаров** — Руководитель тюменского отделения",
                parse_mode="Markdown"
            )
            await query.edit_message_text(text="Фото руководителей отправлены.")
        except Exception as e:
            logger.error(f"Ошибка отправки фотографий: {e}")
            await query.edit_message_text(text="Не удалось отправить фотографии руководителей.")
    elif data == "help":
        prompt = "Предоставь справочную информацию для пользователя, включая примеры запросов."
        response = await generate_gpt_response(prompt)
        await query.edit_message_text(text=response)
    else:
        await query.edit_message_text(text="Команда не распознана.")

# --- Основная функция для запуска бота ---
async def main() -> None:
    global global_index, global_chunks
    global_index, global_chunks = await load_and_index_base_async()
    if global_index is not None and global_chunks:
        logger.info("База знаний загружена и индексирована.")
    else:
        logger.warning("База знаний не загружена. Бот будет работать без доступа к базе знаний.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # (Для отладки можно включить логирование всех обновлений)
    # application.add_handler(MessageHandler(filters.ALL, log_update_handler))
    
    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command_handler))
    application.add_handler(CommandHandler("stop", stop_command_handler))
    
    # Регистрируем обработчики inline-кнопок
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # Регистрируем обработчик для голосовых сообщений и аудиофайлов
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message_handler))
    # Регистрируем обработчик для видео заметок
    application.add_handler(MessageHandler(filters.VIDEO_NOTE, video_note_handler))
    
    # Регистрируем обработчик для текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT, text_message_handler))
    
    logger.info("Бот запущен и готов к работе.")
    application.run_polling()

if __name__ == '__main__':
    # Чтобы избежать ошибки "Cannot close a running event loop", переопределяем временно loop.close
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    original_close = loop.close
    loop.close = lambda: None
    try:
        asyncio.run(main())
    finally:
        loop.close = original_close
