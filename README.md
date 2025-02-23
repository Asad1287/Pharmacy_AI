# Pharmacy_AI
Medical Prescription Analysis System
A comprehensive system for analyzing medical prescriptions using OCR, NER, and LLM technologies. This system extracts text from prescription images, identifies medications and dosage instructions, checks for drug interactions, and provides intelligent analysis using state-of-the-art language models.
Features
1. Optical Character Recognition (OCR)

GPU-accelerated text extraction using both EasyOCR and Tesseract
Advanced image preprocessing including deskewing and enhancement
Support for multiple languages and medical terminology
Confidence scoring for extracted text

2. Named Entity Recognition (NER)

Identification of medications, dosages, frequencies, and routes
Integration with OpenFDA for medication validation
Pattern-based recognition of medical terminology
Automated entity categorization and validation

3. Drug Interaction Analysis

Real-time drug interaction checking via OpenFDA API
Comprehensive adverse event reporting
Analysis of medication combinations
Safety alerts and warnings

4. LLM-Powered Analysis

Intelligent prescription interpretation using Mistral-7B
Context-aware medical recommendations
Patient-friendly instruction generation
Medical terminology explanation

Project Structure
Copyprescription-analysis/
├── Makefile
├── requirements.txt
├── .env
├── scripts/
│   ├── install_dependencies.sh
│   └── install_tesseract.sh
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── ocr_processor.py
│   ├── medication_ner.py
│   ├── drug_interaction.py
│   └── prompt_llm.py
├── tests/
│   └── __init__.py
├── prescriptions_output/
└── logs/
Installation

Clone the repository:

bashCopygit clone <repository-url>
cd prescription-analysis

Create and configure environment file:

bashCopycp .env.example .env
# Edit .env with your OpenFDA API key

Install the system:

bashCopymake all
This will:

Set up Python virtual environment
Install system dependencies
Install Tesseract OCR
Download required ML models
Configure GPU support (if available)

Usage
Basic Usage

Process a single prescription:

bashCopymake run

Run specific components:

bashCopymake run-ocr     # OCR only
make run-ner     # NER only
make run-drug-interaction  # Drug interaction checking
make run-llm     # LLM analysis
Advanced Usage

Process with custom settings:

pythonCopyfrom src.ocr_processor import OCRProcessor
from src.medication_ner import MedicationNER
from src.drug_interaction import DrugInteractionChecker
from src.prompt_llm import PrescriptionPromptEngine

# Initialize processors
ocr = OCRProcessor(lang='en', use_gpu=True)
ner = MedicationNER(api_key='your_openfda_api_key')
interaction_checker = DrugInteractionChecker(api_key='your_openfda_api_key')
prompt_engine = PrescriptionPromptEngine()

# Process prescription
ocr_results = ocr.process_image("prescription.jpg")
ner_results = ner.process_prescription(ocr_results['text'])
interactions = interaction_checker.get_drug_events(ner_results['medications'])
Configuration
Key configuration options in .env:
envCopyOPENFDA_API_KEY=your_api_key_here
USE_GPU=true
OCR_CONFIDENCE_THRESHOLD=0.3
SAVE_INTERMEDIATE_RESULTS=true
Development

Setup development environment:

bashCopymake dev-setup

Run tests:

bashCopymake test

Code quality:

bashCopymake lint
make format
Future Improvements
Short-term Improvements

Enhanced OCR Accuracy

Implement specialized medical text recognition
Add support for handwritten prescriptions
Improve preprocessing for low-quality images


Extended NER Capabilities

Add support for more medical entities
Implement custom medical terminology models
Improve entity relationship detection


Improved Drug Interaction Analysis

Add support for more drug databases
Implement real-time monitoring
Add severity classification for interactions



Long-term Goals

Advanced LLM Integration

Fine-tune models on medical data
Implement multi-modal analysis
Add support for medical image understanding


System Expansion

Add support for electronic health records
Implement patient history tracking
Add support for multiple languages


User Interface

Develop web interface
Create mobile application
Add real-time processing capabilities



Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

OpenFDA for medication data
EasyOCR and Tesseract teams
Mistral AI for the LLM model
All contributors and maintainers

Support
For support, please open an issue in the GitHub repository or contact the maintainers.