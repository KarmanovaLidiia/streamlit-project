#!/usr/bin/env python3
"""
Упрощенный запуск предсказания
"""
import os
import sys

# Устанавливаем PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def main():
    print("🚀 ЗАПУСК ПРЕДСКАЗАНИЯ")

    # Импортируем после установки PYTHONPATH
    from src.predict import pipeline_infer

    # Запускаем предсказание
    input_file = "data/raw/small.csv"
    output_file = "predictions_final.csv"

    print(f"📁 Входной файл: {input_file}")
    print(f"📁 Выходной файл: {output_file}")

    pipeline_infer(input_file, output_file)
    print("🎉 ПРЕДСКАЗАНИЕ ЗАВЕРШЕНО!")


if __name__ == "__main__":
    main()