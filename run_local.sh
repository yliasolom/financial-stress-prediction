#!/bin/bash

# Скрипт для локального запуска API и Streamlit UI

echo "🚀 Запуск локального сервера..."

# Активируем виртуальное окружение
source venv/bin/activate

# Запускаем FastAPI сервер в фоне
echo "📡 Запуск FastAPI сервера на http://localhost:8000"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Ждем немного, чтобы API успел запуститься
sleep 3

# Запускаем Streamlit
echo "🎨 Запуск Streamlit UI на http://localhost:8501"
streamlit run app_ui.py

# При завершении убиваем процесс API
trap "kill $API_PID" EXIT

