import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from src.ocr_processor import GPUAcceleratedOCR
from src.medication_ner import MedicationNER
from src.drug_interaction import DrugInteractionChecker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PrescriptionAnalysisSystem:
    def __init__(self):
        """Initialize the prescription analysis system"""
        self.api_key = os.getenv('OPENFDA_API_KEY')
        if not self.api_key:
            raise ValueError("OpenFDA API key not found in environment variables")

        # Initialize components
        self.ocr = GPUAcceleratedOCR(lang='eng')
        self.ner = MedicationNER(self.api_key)
        self.interaction_checker = DrugInteractionChecker(self.api_key)

    def process_prescription(self, image_path):
        """Process a prescription image through OCR, NER, and drug interaction checking"""
        try:
            logger.info(f"Processing prescription image: {image_path}")

            # Step 1: OCR Processing
            ocr_results = self.ocr.image_to_text(image_path)
            if not ocr_results['text']:
                raise ValueError("No text extracted from image")

            # Step 2: NER Processing
            ner_results = self.ner.process_and_format(ocr_results['text'])

            # Step 3: Drug Interaction Analysis
            medications = self.ner.extract_medications_from_text(ocr_results['text'])
            interaction_results = {}
            for med in medications:
                interaction_results[med] = self.interaction_checker.get_drug_events(med)

            # Save results
            self._save_results(image_path, ocr_results, ner_results, interaction_results)

            return {
                'ocr': ocr_results,
                'ner': ner_results,
                'interactions': interaction_results
            }

        except Exception as e:
            logger.error(f"Error processing prescription: {str(e)}")
            raise

    def _save_results(self, image_path, ocr_results, ner_results, interaction_results):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'prescriptions_output'
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_{timestamp}_analysis.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("PRESCRIPTION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # OCR Results
            f.write("OCR RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Extracted Text:\n{ocr_results['text']}\n")
            f.write(f"Confidence: {ocr_results['confidence']}%\n\n")

            # NER Results
            f.write("ENTITY RECOGNITION RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(ner_results + "\n")

            # Drug Interaction Results
            f.write("DRUG INTERACTION ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            for med, data in interaction_results.items():
                f.write(f"\nMedication: {med}\n")
                if 'error' in data:
                    f.write(f"Error: {data['error']}\n")
                else:
                    f.write(f"Total Reports: {data.get('total_reports', 0)}\n")
                    if 'reactions' in data:
                        f.write("Top Reactions:\n")
                        for reaction, count in list(data['reactions'].items())[:5]:
                            f.write(f"- {reaction}: {count} reports\n")

        logger.info(f"Analysis results saved to: {output_path}")
        return output_path

def main():
    """Main execution function"""
    try:
        # Initialize the system
        system = PrescriptionAnalysisSystem()

        # Process test prescription
        test_image = "test_prescription.jpg"
        if not os.path.exists(test_image):
            logger.error(f"Test image not found: {test_image}")
            return

        # Process the prescription
        results = system.process_prescription(test_image)
        logger.info("Prescription processing completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()