import requests

url = "http://localhost:8000/predict_csv"
file_path = r".\data\raw\Данные для кейса.csv"

print(f"📤 Отправляю файл {file_path} на {url}...")

with open(file_path, 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    response = requests.post(url, files=files, timeout=120)  # 2 минуты таймаут

print(f"📥 Ответ: {response.status_code}")

if response.status_code == 200:
    with open('predicted_from_api.csv', 'wb') as f:
        f.write(response.content)
    print("✅ Файл успешно обработан! Результат в predicted_from_api.csv")
else:
    print(f"❌ Ошибка: {response.status_code}")
    print("Текст ответа:", response.text)