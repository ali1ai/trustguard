.PHONY: install install-training test lint run train-spam train-toxicity
install:
	python -m pip install -e '.[dev]'
install-training:
	python -m pip install -e '.[dev,training]'
test:
	pytest -q
lint:
	ruff check src tests scripts
run:
	uvicorn trustguard.api.main:app --reload
train-spam:
	python scripts/download_sms_dataset.py
	python scripts/train_spam_classifier.py
train-toxicity:
	python scripts/train_toxicity_classifier.py
