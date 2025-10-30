.PHONY: train predict api ui test docker

predict:
	python -m src.predict --input data/raw/Данные\ для\ кейса.csv --output data/processed/predicted.csv

api:
	uvicorn app.main:app --host 127.0.0.1 --port 8020 --reload

ui:
	streamlit run app/ui.py

test:
	pytest -q

docker:
	docker compose up --build
