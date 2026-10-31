# Telco Retention Decision Engine (TRDE) Makefile
.PHONY: help setup pipeline test serve streamlit clean

help:
	@echo "Telco Retention Decision Engine Commands:"
	@echo "  make setup     - Install lightweight C++/Rust dependencies"
	@echo "  make pipeline  - Run end-to-end lakehouse, modeling, MILP optimizer & MLflow logging"
	@echo "  make test      - Execute complete pytest suite across all layers"
	@echo "  make serve     - Launch FastAPI decision service on http://127.0.0.1:8000"
	@echo "  make streamlit - Launch Streamlit decision console on http://127.0.0.1:8501"
	@echo "  make clean     - Clean temporary artifacts and cache directories"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

pipeline:
	python src/pipeline.py

test:
	pytest -v

serve:
	uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

streamlit:
	streamlit run streamlit_app.py

clean:
	python -c "import shutil, os, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True)]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
