import requests

url = "http://localhost:8000/predict_csv"
file_path = "small.csv"  # положи файл в корень проекта

print(f"📤 Отправляю файл {file_path}...")
with open(file_path, 'rb') as f:
    files = {'file': ('small.csv', f, 'text/csv')}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open('predicted_small.csv', 'wb') as f:
        f.write(response.content)
    print("✅ Файл обработан! predicted_small.csv")
else:
    print(f"❌ Ошибка: {response.status_code}")
    print(response.text)