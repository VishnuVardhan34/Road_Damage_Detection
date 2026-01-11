# Makefile for Road Damage Detection Project

.PHONY: help install train evaluate infer api test clean docker docker-up docker-down

help:
	@echo "Road Damage Detection - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install dev dependencies"
	@echo ""
	@echo "Data Processing:"
	@echo "  make convert-data  - Convert VOC to YOLO format"
	@echo ""
	@echo "Training:"
	@echo "  make train         - Train YOLOv8-L model"
	@echo "  make train-m       - Train YOLOv8-M model"
	@echo ""
	@echo "Inference:"
	@echo "  make infer         - Run inference on samples"
	@echo "  make api           - Start API server"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate      - Evaluate model on test set"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run unit tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo ""
	@echo "Docker:"
	@echo "  make docker        - Build Docker image"
	@echo "  make docker-up     - Start Docker Compose"
	@echo "  make docker-down   - Stop Docker Compose"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean up generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort

convert-data:
	python scripts/convert_dataset.py \
		--voc_dir data/raw/annotations \
		--output_dir data/processed/labels

train:
	python scripts/train.py \
		--config config/train_config.yaml \
		--variant l

train-m:
	python scripts/train.py \
		--config config/train_config.yaml \
		--variant m

evaluate:
	python scripts/evaluate.py \
		--model checkpoints/best_model.pt \
		--data data/processed \
		--output results/evaluation.json

infer:
	python scripts/infer.py \
		--model checkpoints/best_model.pt \
		--dir data/processed/images/test \
		--output results/inference

api:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

docker:
	docker build -f docker/Dockerfile -t road-damage-detection:latest .

docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov *.egg-info dist build
	rm -rf logs/* results/*

format:
	black src/ scripts/
	isort src/ scripts/

lint:
	flake8 src/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics

notebook-eda:
	jupyter notebook notebooks/01_EDA.ipynb

notebook-augmentation:
	jupyter notebook notebooks/02_Data_Augmentation.ipynb

notebook-results:
	jupyter notebook notebooks/03_Results_Analysis.ipynb
