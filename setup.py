import os
import sys
import subprocess


def setup_environment():
    """Устанавливает PYTHONPATH и возвращает команду для запуска"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Добавляем в PYTHONPATH
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Устанавливаем переменную окружения для дочерних процессов
    os.environ['PYTHONPATH'] = current_dir + os.pathsep + os.environ.get('PYTHONPATH', '')

    print(f"✅ PYTHONPATH установлен: {current_dir}")
    return current_dir


if __name__ == "__main__":
    setup_environment()

    # Теперь можно запускать predict.py
    print("🚀 Запуск predict.py...")
    try:
        from src.predict import pipeline_infer

        pipeline_infer("data/raw/small.csv", "predictions.csv")
        print("✅ Предсказание завершено!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")