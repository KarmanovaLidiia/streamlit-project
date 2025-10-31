from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import csv
import os
import tempfile
from typing import List, Dict
import re

app = FastAPI(title="Russian Exam Auto Grader")

# Монтируем статические файлы для веб-интерфейса
app.mount("/static", StaticFiles(directory="static"), name="static")


class ExamGrader:
    def __init__(self):
        self.setup_criteria()

    def setup_criteria(self):
        self.criteria = {
            1: self._grade_question1,  # 0-1 балл
            2: self._grade_question2,  # 0-2 балла
            3: self._grade_question3,  # 0-1 балл
            4: self._grade_question4  # 0-2 балла
        }

    def grade_answer(self, question_num: int, transcription: str) -> int:
        """Основной метод оценки"""
        if question_num not in self.criteria:
            return 0
        return self.criteria[question_num](transcription)

    def _grade_question1(self, text: str) -> int:
        """Оценка вопроса 1 - начало диалога"""
        text_lower = text.lower().strip()

        # Проверяем ключевые элементы диалога
        has_greeting = any(word in text_lower for word in ['здравствуйте', 'добрый день', 'привет', 'здравствуй'])
        has_request = any(word in text_lower for word in ['помогите', 'подскажите', 'нужно', 'хочу', 'могу'])
        has_question = any(word in text_lower for word in ['как', 'что', 'где', 'когда', 'можно', 'сколько'])

        # Должен быть развернутый ответ
        words_count = len(text_lower.split())

        score = 0
        if has_greeting:
            score += 0.3
        if has_request:
            score += 0.4
        if has_question:
            score += 0.3
        if words_count > 15:
            score += 0.2

        return 1 if score >= 0.7 else 0

    def _grade_question2(self, text: str) -> int:
        """Оценка вопроса 2 - ответы на вопросы"""
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return 0

        # Оцениваем полноту ответов
        complete_sentences = 0
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 4:  # Более-менее полное предложение
                complete_sentences += 1

        completeness_ratio = complete_sentences / len(sentences)

        if completeness_ratio >= 0.8:
            return 2
        elif completeness_ratio >= 0.5:
            return 1
        else:
            return 0

    def _grade_question3(self, text: str) -> int:
        """Оценка вопроса 3 - диалог-запрос"""
        text_lower = text.lower().strip()

        has_greeting = any(word in text_lower for word in ['здравствуйте', 'добрый день'])
        has_request = any(word in text_lower for word in ['хочу', 'нужно', 'узнать', 'скажите', 'интересует'])
        has_thanks = any(word in text_lower for word in ['спасибо', 'благодарю'])

        score = 0
        if has_greeting:
            score += 0.3
        if has_request:
            score += 0.4
        if has_thanks:
            score += 0.3

        return 1 if score >= 0.7 else 0

    def _grade_question4(self, text: str) -> int:
        """Оценка вопроса 4 - описание картинки"""
        sentences = self._split_sentences(text)

        if len(sentences) < 3:
            return 0

        # Ищем описательные элементы
        descriptive_words = ['вижу', 'изображен', 'находится', 'стоит', 'сидит',
                             'одежда', 'цвет', 'время года', 'место', 'деревья', 'дом']

        descriptive_count = 0
        for sentence in sentences:
            if any(word in sentence.lower() for word in descriptive_words):
                descriptive_count += 1

        descriptive_ratio = descriptive_count / len(sentences)

        if descriptive_ratio >= 0.6:
            return 2
        elif descriptive_ratio >= 0.3:
            return 1
        else:
            return 0

    def _split_sentences(self, text: str) -> List[str]:
        """Разделяет текст на предложения"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]


grader = ExamGrader()


@app.post("/evaluate/")
async def evaluate_file(file: UploadFile = File(...)):
    try:
        # Читаем CSV файл
        content = await file.read()
        decoded_content = content.decode('utf-8').splitlines()

        # Парсим CSV
        reader = csv.DictReader(decoded_content, delimiter=';')
        rows = list(reader)

        # Обрабатываем каждую строку
        results = []
        for row in rows:
            try:
                question_num = int(row['№ вопроса'])
                transcription = row['Транскрибация ответа']

                score = grader.grade_answer(question_num, transcription)

                result_row = row.copy()
                result_row['Оценка экзаменатора'] = score
                results.append(result_row)
            except (KeyError, ValueError) as e:
                # Если есть ошибки в данных, ставим 0
                result_row = row.copy()
                result_row['Оценка экзаменатора'] = 0
                results.append(result_row)

        # Сохраняем результаты
        output_filename = "graded_" + file.filename
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                writer.writerows(results)

        return FileResponse(
            output_filename,
            media_type='text/csv',
            filename=output_filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <head>
            <title>Russian Exam Auto Grader</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 600px; margin: 0 auto; }
                .upload-form { border: 2px dashed #ccc; padding: 40px; text-align: center; }
                .btn { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
                .btn:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Russian Exam Auto Grader</h1>
                <p>Загрузите CSV файл с ответами для автоматической оценки</p>

                <form class="upload-form" action="/evaluate/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".csv" required>
                    <br><br>
                    <button type="submit" class="btn">Оценить ответы</button>
                </form>

                <div style="margin-top: 30px;">
                    <h3>Требования к файлу:</h3>
                    <ul>
                        <li>Формат: CSV с разделителем ";"</li>
                        <li>Колонки: № вопроса, Транскрибация ответа</li>
                        <li>Кодировка: UTF-8</li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)