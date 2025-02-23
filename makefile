# Makefile for Prescription Analysis System

# Python variables
PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python

# Directories
SRC_DIR = src
TEST_DIR = tests
OUTPUT_DIR = prescriptions_output
MODELS_DIR = models
LOGS_DIR = logs

# Files
ENV_FILE = .env
REQUIREMENTS = requirements.txt

.PHONY: all setup install clean run test lint format help

# Default target
all: setup install download-models

# Create and setup virtual environment
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing requirements..."
	$(PIP) install -r $(REQUIREMENTS)

# Install system dependencies
install:
	@echo "Installing system dependencies..."
	bash scripts/install_dependencies.sh
	@echo "Installing Tesseract..."
	bash scripts/install_tesseract.sh

# Download required models
download-models:
	@echo "Creating directories..."
	mkdir -p $(OUTPUT_DIR) $(MODELS_DIR) $(LOGS_DIR)
	@echo "Downloading spaCy model..."
	$(PYTHON_VENV) -m spacy download en_core_web_sm
	@echo "Downloading EasyOCR models..."
	$(PYTHON_VENV) -c "import easyocr; reader = easyocr.Reader(['en'])"

# Clean up generated files and virtual environment
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -rf $(OUTPUT_DIR)
	rm -rf $(MODELS_DIR)
	rm -rf $(LOGS_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	rm -rf drug_safety_report.txt
	rm -rf medication_analysis_results.txt
	rm -rf ocr_results.txt

# Run the prescription analysis system
run:
	@echo "Running prescription analysis system..."
	$(PYTHON_VENV) $(SRC_DIR)/main.py

# Run specific components
run-ocr:
	@echo "Running OCR processor..."
	$(PYTHON_VENV) $(SRC_DIR)/ocr_processor.py

run-ner:
	@echo "Running NER analysis..."
	$(PYTHON_VENV) $(SRC_DIR)/medication_ner.py

run-drug-interaction:
	@echo "Running drug interaction analysis..."
	$(PYTHON_VENV) $(SRC_DIR)/drug_interaction.py

run-llm:
	@echo "Running LLM analysis..."
	$(PYTHON_VENV) $(SRC_DIR)/prompt_llm.py

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON_VENV) -m pytest $(TEST_DIR)/

# Code quality
lint:
	@echo "Running linter..."
	$(PYTHON_VENV) -m flake8 $(SRC_DIR)
	$(PYTHON_VENV) -m flake8 $(TEST_DIR)

format:
	@echo "Formatting code..."
	$(PYTHON_VENV) -m black $(SRC_DIR)
	$(PYTHON_VENV) -m black $(TEST_DIR)

# Check environment setup
check-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "Error: $(ENV_FILE) not found. Please create it with your API keys."; \
		exit 1; \
	fi

# Help target
help:
	@echo "Available targets:"
	@echo "  all              : Setup environment and install all dependencies"
	@echo "  setup           : Create virtual environment and install Python packages"
	@echo "  install         : Install system dependencies"
	@echo "  download-models : Download required ML models"
	@echo "  clean           : Remove all generated files and virtual environment"
	@echo "  run             : Run the complete prescription analysis system"
	@echo "  run-ocr         : Run only the OCR processor"
	@echo "  run-ner         : Run only the NER analysis"
	@echo "  run-drug-interaction : Run only the drug interaction analysis"
	@echo "  run-llm         : Run only the LLM analysis"
	@echo "  test            : Run tests"
	@echo "  lint            : Run code linter"
	@echo "  format          : Format code"
	@echo "  help            : Show this help message"

# Create initial project structure
init:
	@echo "Creating project structure..."
	mkdir -p $(SRC_DIR)
	mkdir -p $(TEST_DIR)
	mkdir -p $(OUTPUT_DIR)
	mkdir -p $(MODELS_DIR)
	mkdir -p $(LOGS_DIR)
	mkdir -p scripts
	touch $(ENV_FILE)
	@echo "Project structure created. Please update $(ENV_FILE) with your API keys."

# Environment setup check
check-gpu:
	@echo "Checking GPU availability..."
	$(PYTHON_VENV) -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Install additional development dependencies
dev-setup: setup
	@echo "Installing development dependencies..."
	$(PIP) install black flake8 pytest pytest-cov

# Default target when no arguments provided
.DEFAULT_GOAL := help