# deploy-to-yandex.ps1
Write-Host "🚀 Начало развертывания в Yandex Cloud..." -ForegroundColor Green

# Переменные (ЗАМЕНИТЕ на свои!)
$REGISTRY_ID = "your-registry-id"  # Найти в консоли: Container Registry -> ID реестра
$IMAGE_NAME = "exam-scorer"
$TAG = "latest"
$FULL_IMAGE = "cr.yandex/$REGISTRY_ID/$IMAGE_NAME`:$TAG"

# 1. Сборка Docker образа
Write-Host "📦 Сборка Docker образа..." -ForegroundColor Yellow
docker build -t $FULL_IMAGE .

# 2. Авторизация в Yandex Container Registry
Write-Host "🔐 Авторизация в Container Registry..." -ForegroundColor Yellow
yc container registry configure-docker

# 3. Загрузка образа в реестр
Write-Host "⬆️ Загрузка образа в Yandex Cloud..." -ForegroundColor Yellow
docker push $FULL_IMAGE

Write-Host "✅ Образ успешно загружен: $FULL_IMAGE" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Дальнейшие действия:" -ForegroundColor Cyan
Write-Host "1. В консоли Yandex Cloud перейдите в 'Serverless Containers'"
Write-Host "2. Создайте новый контейнер"
Write-Host "3. Укажите образ: $FULL_IMAGE"
Write-Host "4. Настройте порт: 8000"
Write-Host "5. Задайте переменные окружения:"
Write-Host "   - PYTHONPATH=/app"