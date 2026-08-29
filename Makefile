.PHONY: install test lint demo benchmark serve docker

install:
	python -m pip install -e '.[api,dev]'

test:
	PYTHONPATH=src python -m pytest -q

lint:
	python -m ruff check src tests

demo:
	PYTHONPATH=src python -m pdserve.cli demo

benchmark:
	PYTHONPATH=src python -m pdserve.cli benchmark

serve:
	PYTHONPATH=src python -m pdserve.cli serve

docker:
	docker compose up --build
