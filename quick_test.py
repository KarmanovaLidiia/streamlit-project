### **3. `quick_test.py`** (быстрая проверка)
```python
# !/usr/bin/env python3
"""
Быстрая проверка работы системы
"""

import subprocess
import sys
import os


def run_command(cmd):
    """Запускает команду и возвращает результат"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def main():
    print("🚀 БЫСТРАЯ ПРОВЕРКА СИСТЕМЫ")
    print("=" * 50)

    # 1. Проверяем зависимости
    print("1. Проверка зависимостей...")
    success, out, err = run_command(
        "python -c \"import catboost, fastapi, streamlit; print('✅ Все зависимости установлены')\"")
    if success:
        print("   ✅ Все зависимости установлены")
    else:
        print("   ❌ Ошибка зависимостей:", err)
        return

    # 2. Проверяем модели
    print("2. Проверка ML моделей...")
    models = ["catboost_Q1.cbm", "catboost_Q2.cbm", "catboost_Q3.cbm", "catboost_Q4.cbm"]
    all_models_exist = all(os.path.exists(f"models/{model}") for model in models)
    if all_models_exist:
        print("   ✅ Все ML модели найдены")
    else:
        print("   ❌ Не все модели найдены")
        return

    # 3. Проверяем данные
    print("3. Проверка данных...")
    if os.path.exists("data/raw/small.csv"):
        print("   ✅ Тестовые данные найдены")
    else:
        print("   ⚠️ Тестовые данные не найдены")

    print("\n🎉 СИСТЕМА ГОТОВА К РАБОТЕ!")
    print("Запустите: docker-compose up")


if __name__ == "__main__":
    main()