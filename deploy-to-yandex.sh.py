#!/bin/bash
set -e

echo "🚀 Начало развертывания в Yandex Cloud..."

# Переменные (замените на свои)
REGISTRY_ID="your-registry-id"  # Найти в консоли: Container Registry -> ID реестра
IMAGE_NAME="exam-scorer"
TAG="latest"
FULL_IMAGE="cr.yandex/${REGISTRY_ID}/${IMAGE_NAME}:${TAG}"

# 1. Сборка Docker образа
echo "📦 Сборка Docker образа..."
docker build -t ${FULL_IMAGE} .

# 2. Авторизация в Yandex Container Registry
echo "🔐 Авторизация в Container Registry..."
yc container registry configure-docker

# 3. Загрузка образа в реестр
echo "⬆️ Загрузка образа в Yandex Cloud..."
docker push ${FULL_IMAGE}

echo "✅ Образ успешно загружен: ${FULL_IMAGE}"
echo ""
echo "🎯 Дальнейшие действия:"
echo "1. В консоли Yandex Cloud перейдите в 'Serverless Containers'"
echo "2. Создайте новый контейнер"
echo "3. Укажите образ: ${FULL_IMAGE}"
echo "4. Настройте порт: 8000"
echo "5. Задайте переменные окружения:"
echo "   - PYTHONPATH=/app"