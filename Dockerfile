FROM python:3.12-slim

# Установка системных зависимостей для OpenCV и других библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt /tmp/requirements.txt

# Обновляем pip, setuptools и wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY . /app/

CMD ["python", "inference.py"]