# GARUDA-NBA Makefile for Windows / POSIX environments
.PHONY: help setup pipeline test serve clean

help:
	@echo "GARUDA-NBA Automation Commands:"
	@echo "  make setup     - Install lightweight C++/Rust dependencies (< 400MB RAM)"
	@echo "  make pipeline  - Run end-to-end lakehouse, modeling, MILP optimizer & MLflow logging"
	@echo "  make test      - Execute complete pytest suite across all layers"
	@echo "  make serve     - Launch FastAPI decision service on http://127.0.0.1:8000"
	@echo "  make clean     - Clean temporary artifacts and cache directories"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-8gb.txt

pipeline:
	python src/pipeline.py

test:
	pytest -v

serve:
	uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

clean:
	python -c "import shutil, os, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
