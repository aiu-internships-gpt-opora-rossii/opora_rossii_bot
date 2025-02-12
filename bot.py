import logging
import os
import re
import asyncio
import time
import csv
import io
from dotenv import load_dotenv
import nest_asyncio

import openai
import faiss
import numpy as np

# LangChain для более точного разбиения текста на смысловые чанки
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Для конвертации аудио форматов
from pydub import AudioSegment

# Импорт функции получения эмбеддингов из нового API OpenAI
from openai.embeddings_utils import get_embedding

# Импорт локальной модели Whisper для транскрибации
import whisper

# Загружаем модель Whisper "large" (если нужна другая, например, "Whisper Real Time", измените параметр)
WHISPER_MODEL = whisper.load_model("large")
logging.getLogger(__name__).info("Whisper модель 'large' загружена.")

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

# Проверка переменных окружения
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

# Глобальная переменная для хранения истории диалога
conversation_history = []

# --- Параметры для FAISS и LangChain ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5  # количество возвращаемых чанков при поиске

# URL-ы фотографий руководителей (пример)
PHOTO_PRESIDENT = "https://static.tildacdn.com/tild6139-3564-4162-a636-613162376438/ca38d0e45e485eeaa214.jpg"
PHOTO_VP_URL = "https://static.tildacdn.com/tild3033-3264-4032-a362-393764633238/12312312314444444444.jpg"

# Пример дополнительных текстовых данных
MEMBERSHIP_ADVANTAGES = (
    "• Развитие профессиональных навыков: участие в образовательных программах, мастер-классах и вебинарах.\n"
    "• Расширение деловых связей: обмен опытом с единомышленниками и поиск партнёров.\n"
    "• Юридическая и административная поддержка: бесплатные консультации юристов и доступ к шаблонам документов.\n"
    "• Индивидуальные консультации: личный разбор бизнес-задач.\n"
    "• Экономические выгоды: скидки и эксклюзивные предложения от партнёров."
)

# --- Функции для работы с базой знаний ---

def convert_docx_to_txt(input_filename: str = "russia_base.docx", output_filename: str = "russia_base.txt") -> bool:
    """
    При отсутствии russia_base.txt пытаемся сконвертировать russia_base.docx в txt.
    """
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
        logger.info(f"Файл {input_filename} сконвертирован в {output_filename}.")
        return True
    except Exception as e:
        logger.error(f"Ошибка при конвертации: {e}")
        return False

async def split_text_into_meaningful_chunks(text: str) -> list:
    """
    Разбиваем текст на смысловые чанки с помощью LangChain RecursiveCharacterTextSplitter.
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # пробуем абзацы, предложения и т.д.
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Разбито на {len(chunks)} чанков (LangChain).")
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
            logger.error(f"Ошибка эмбеддинга (попытка {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                return np.zeros(1536, dtype="float32")
    return np.zeros(1536, dtype="float32")

async def load_and_index_base_async(filename: str = "russia_base.txt", docx_filename: str = "russia_base.docx"):
    """
    Загружаем (или конвертируем) файл базы знаний, разбиваем с помощью LangChain
    и создаём FAISS-индекс.
    """
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
        logger.error(f"Ошибка чтения базы знаний: {e}")
        return faiss.IndexFlatL2(1536), []
    
    # Разбиваем текст на осмысленные чанки с помощью LangChain
    chunks = await split_text_into_meaningful_chunks(text)

    # Вычисляем эмбеддинги чанков
    embeddings = await asyncio.gather(*(get_embedding_for_chunk(chunk) for chunk in chunks))
    embeddings = np.stack(embeddings)
    dim = embeddings.shape[1]

    # Создаём FAISS-индекс
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"FAISS индекс создан с {index.ntotal} элементами.")
    return index, chunks

# Функция получения актуальной информации о мероприятиях из Google Sheets
def fetch_events_from_sheet():
    csv_url = "https://docs.google.com/spreadsheets/d/1Ow7kGA9TdgpTaq2ABhy700ApKApMd_4kZoXfU7oqcm4/export?format=csv&gid=0"
    try:
        import requests
        response = requests.get(csv_url)
        response.raise_for_status()
        csv_data = response.text
        reader = csv.reader(io.StringIO(csv_data))
        events = []
        for row in reader:
            if len(row) >= 2:
                event_name = row[0].strip()
                event_link = row[1].strip()
                events.append(f"{event_name}: {event_link}")
        events_text = "\n".join(events)
        logger.info("Информация о мероприятиях из Google Sheets:\n%s", events_text)
        return events_text
    except Exception as e:
        logger.error(f"Ошибка загрузки данных Google Sheets: {e}")
        return "Информация о мероприятиях временно недоступна."

# --- Поиск контекста в базе знаний ---
async def retrieve_context(query: str, k: int = TOP_K) -> str:
    """
    Ищем top-k наиболее релевантных чанков. k=TOP_K (по умолчанию 5).
    """
    if global_index is None or not global_chunks:
        return ""
    loop = asyncio.get_running_loop()
    query_embedding = await loop.run_in_executor(
        None, lambda: np.array(get_embedding(query, engine="text-embedding-ada-002"), dtype="float32")
    )
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = global_index.search(query_embedding, k)
    result_chunks = []
    for i in indices[0]:
        if i < len(global_chunks):
            result_chunks.append(global_chunks[i])
    logger.info("Top-%d чанки:\n%s", k, "\n-----\n".join(result_chunks))
    return "\n".join(result_chunks)

# --- Генерация ответа через GPT-4 ---
async def generate_gpt_response(prompt: str) -> str:
    # Если вопрос про Константина Константинова
    if "константин константинов" in prompt.lower():
        override_context = (
            "Председатели местных отделений ТюмРО «ОПОРА РОССИИ»:\n"
            "1. МО г.Тобольск — Константинов Константин Юрьевич"
        )
        full_prompt = f"{override_context}\n\nОтветь на запрос: {prompt}"
    else:
        # Подгружаем контекст из базы знаний
        context_text = await retrieve_context(prompt, k=TOP_K)
        full_prompt = f"Используй информацию из базы знаний:\n{context_text}\n\nОтветь на запрос: {prompt}"
    
    logger.info("Полный промпт GPT-4:\n%s", full_prompt)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        'Ты — Telegram-бот "Опора России". '
                        'Правильные данные: Президент — Александр Калинин; '
                        'Первый вице-президент и руководитель тюменского отделения — Эдуард Омаров. '
                        'Если упоминается Константин Константинов, он председатель МО г.Тобольск. '
                        'Отвечай полно, информативно, опираясь на базу знаний и актуальные материалы.'
                    )
                },
                {"role": "user", "content": full_prompt}
            ],
            temperature=0,
            max_tokens=2000,
        )
        text = response.choices[0].message["content"].strip()
        logger.info("Ответ от GPT-4:\n%s", text)
        return text
    except Exception as e:
        logger.error(f"Ошибка вызова OpenAI: {e}")
        return "Извините, произошла ошибка при формировании ответа."

# --- Обработчики команд и сообщений ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    conversation_history.clear()
    logger.info(f"Пользователь {user.username} ({user.id}) ввёл /start.")
    welcome_text = (
        "Здравствуйте!\n\n"
        "Добро пожаловать в бот «Опора России». "
        "Здесь вы можете узнать информацию о мероприятиях, услугах и преимуществах членства.\n\n"
        "Доступные команды:\n"
        "/start — запуск бота\n"
        "/help — справка\n"
        "/stop — остановить бота (только админ)\n"
        "/getid — узнать свой ID\n\n"
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
    conversation_history.append({"role": "system", "content": welcome_text})

async def help_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Пользователь {user.username} ({user.id}) ввёл /help.")
    help_text = (
        "📚 **Помощь** 📚\n\n"
        "**Команды:**\n"
        "/start — запуск бота\n"
        "/help — справка\n"
        "/stop — остановить бота (админ)\n"
        "/getid — узнать Telegram ID\n\n"
        "**Примеры:**\n"
        "— Кто сейчас возглавляет Опору России?\n"
        "— Кто руководит тюменским отделением?\n"
        "— Расскажи о преимуществах членства."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")
    conversation_history.append({"role": "system", "content": help_text})

async def stop_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user.id != ADMIN_USER_ID:
        await update.message.reply_text("У вас нет прав для этой команды.")
        logger.warning(f"Пользователь {user.username} ({user.id}) без прав ввёл /stop.")
        return
    logger.info(f"Пользователь {user.username} ({user.id}) ввёл /stop.")
    await update.message.reply_text("🔚 До свидания!")
    conversation_history.append({"role": "system", "content": "Диалог завершён."})
    await context.application.stop()

# Пример обработчика для голосовых/аудио
async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Получено аудио от {user.username} ({user.id}).")
    voice = update.message.voice or update.message.audio
    if not voice:
        await update.message.reply_text("Аудио не распознано.")
        return

    try:
        await update.message.reply_text("Голосовое сообщение обрабатывается...")
        file = await voice.get_file()
        timestamp = int(time.time())
        ogg_path = f"voice_{user.id}_{timestamp}.ogg"
        await file.download_to_drive(ogg_path)
        logger.info(f"Файл сохранен: {ogg_path}")

        # Конвертация в WAV
        audio_seg = AudioSegment.from_file(ogg_path, format="ogg")
        wav_path = ogg_path.replace(".ogg", ".wav")
        audio_seg.export(wav_path, format="wav")
        logger.info(f"Сконвертирован в WAV: {wav_path}")

        # Whisper
        result = WHISPER_MODEL.transcribe(wav_path, language="ru", fp16=False)
        transcription = result["text"].strip()
        transcription = re.sub(r'\bаппора\b', 'Опора', transcription, flags=re.IGNORECASE)
        logger.info(f"Транскрибированный текст: {transcription}")

        os.remove(ogg_path)
        os.remove(wav_path)
        await update.message.reply_text(f"💡 Распознанный текст: {transcription}")
        conversation_history.append({"role": "user", "content": transcription})
        response = await generate_gpt_response(transcription)
        await update.message.reply_text(response)
        conversation_history.append({"role": "assistant", "content": response})
    except Exception as e:
        logger.error(f"Ошибка при обработке аудио: {e}")
        await update.message.reply_text("Ошибка при обработке голосового сообщения.")

# Пример обработчика для видео заметок
async def video_note_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    logger.info(f"Получена видео заметка от {user.username} ({user.id}).")
    try:
        await update.message.reply_text("Видео заметка обрабатывается...")
        video_note = update.message.video_note
        if not video_note:
            await update.message.reply_text("Не удалось получить видео заметку.")
            return

        file = await video_note.get_file()
        timestamp = int(time.time())
        mp4_path = f"video_note_{user.id}_{timestamp}.mp4"
        await file.download_to_drive(mp4_path)
        logger.info(f"Видео заметка: {mp4_path}")

        # Извлекаем аудио
        audio_seg = AudioSegment.from_file(mp4_path, format="mp4")
        wav_path = mp4_path.replace(".mp4", ".wav")
        audio_seg.export(wav_path, format="wav")
        logger.info(f"Аудио извлечено: {wav_path}")

        result = WHISPER_MODEL.transcribe(wav_path, language="ru", fp16=False)
        transcription = result["text"].strip()
        transcription = re.sub(r'\bаппора\b', 'Опора', transcription, flags=re.IGNORECASE)
        logger.info(f"Транскрибированный текст (видео заметка): {transcription}")

        os.remove(mp4_path)
        os.remove(wav_path)

        if transcription:
            await update.message.reply_text(f"💡 Распознанный текст: {transcription}")
            conversation_history.append({"role": "user", "content": transcription})
            response = await generate_gpt_response(transcription)
            await update.message.reply_text(response)
            conversation_history.append({"role": "assistant", "content": response})
        else:
            await update.message.reply_text("Не удалось получить текст.")
    except Exception as e:
        logger.error(f"Ошибка при обработке видео заметки: {e}")
        await update.message.reply_text("Произошла ошибка при обработке видео.")

# Обработчик текстовых сообщений
async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:
        user_text = update.message.text.strip()
        logger.info(f"Текстовое сообщение: {user_text}")
        if user_text.startswith("/"):
            return  # пропускаем команды
        conversation_history.append({"role": "user", "content": user_text})
        response = await generate_gpt_response(user_text)
        await update.message.reply_text(response)
        conversation_history.append({"role": "assistant", "content": response})
    else:
        logger.info("Нет текстового сообщения.")

# Обработчик inline-кнопок
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    logger.info(f"Inline-кнопка: {data}")

    if data == "events":
        events_info = fetch_events_from_sheet()
        prompt = f"Расскажи о предстоящих мероприятиях. Вот информация:\n{events_info}"
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
                caption="**Эдуард Омаров** — Первый вице-президент и руководитель тюменского отделения",
                parse_mode="Markdown"
            )
            await query.edit_message_text(text="Фото руководителей отправлены.")
        except Exception as e:
            logger.error(f"Ошибка отправки фотографий: {e}")
            await query.edit_message_text(text="Не удалось отправить фотографии.")
    elif data == "help":
        prompt = "Предоставь справочную информацию (примеры вопросов)."
        response = await generate_gpt_response(prompt)
        await query.edit_message_text(text=response)
    else:
        await query.edit_message_text(text="Команда не распознана.")

# --- Основная функция запуска бота ---
async def main() -> None:
    global global_index, global_chunks
    # Загружаем и индексируем базу знаний (LangChain + FAISS)
    global_index, global_chunks = await load_and_index_base_async()
    if global_index is not None and global_chunks:
        logger.info("База знаний загружена и индексирована.")
    else:
        logger.warning("База знаний не загружена. Бот работает без базы.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command_handler))
    application.add_handler(CommandHandler("stop", stop_command_handler))
    application.add_handler(CallbackQueryHandler(button_handler))

    # Голосовые/Аудио
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message_handler))
    # Видео заметки
    application.add_handler(MessageHandler(filters.VIDEO_NOTE, video_note_handler))
    # Текст
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    logger.info("Бот запущен и готов к работе.")
    application.run_polling()

if __name__ == '__main__':
    # Чтобы избежать ошибки "Cannot close a running event loop"
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    original_close = loop.close
    loop.close = lambda: None
    try:
        asyncio.run(main())
    finally:
        loop.close = original_close
