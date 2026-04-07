test:
	poetry run pytest tests/

cov:
	poetry run pytest --cov=model_track --cov-report=term-missing --cov-report=xml