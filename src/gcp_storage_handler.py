from google.cloud import storage
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import json
from src.ocr_processor import process_prescription_image
from src.medication_ner import analyze_prescription
from src.drug_interaction import DrugInteractionChecker
from src.prompt_llm import create_prescription_prompt
import concurrent.futures
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCPPrescriptionHandler:
    """
    Complete handler for processing prescriptions using GCP Storage
    """
    
    def __init__(self, 
                project_id: str,
                input_bucket: str,
                output_bucket: str,
                credentials_path: Optional[str] = None,
                batch_size: int = 5,
                max_workers: int = 3):
        """
        Initialize the GCP Prescription Handler
        
        Args:
            project_id: GCP project ID
            input_bucket: Input bucket name
            output_bucket: Output bucket name
            credentials_path: Path to GCP credentials
            batch_size: Number of images to process in parallel
            max_workers: Maximum number of concurrent workers
        """
        self.project_id = project_id
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Set credentials if provided
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
        # Initialize storage client
        self.storage_client = storage.Client()
        self.input_bucket = self.storage_client.bucket(input_bucket)
        self.output_bucket = self.storage_client.bucket(output_bucket)
        
        # Initialize processing directories
        self.base_dir = os.path.join(os.getcwd(), 'gcp_processing')
        self.download_dir = os.path.join(self.base_dir, 'downloads')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
        # Create directories
        for directory in [self.download_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize components
        self.drug_checker = DrugInteractionChecker(os.getenv('OPENFDA_API_KEY'))
        
        logger.info(f"Initialized GCP Prescription Handler for project {project_id}")

    def process_bucket(self, prefix: Optional[str] = None) -> Dict:
        """
        Process all prescriptions in the bucket
        
        Args:
            prefix: Optional prefix to filter images
            
        Returns:
            Processing statistics
        """
        try:
            start_time = time.time()
            stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'processing_time': 0
            }
            
            # List all blobs in bucket
            blobs = list(self.input_bucket.list_blobs(prefix=prefix))
            image_blobs = [
                blob for blob in blobs 
                if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
            ]
            
            logger.info(f"Found {len(image_blobs)} images to process")
            
            # Process in batches
            for i in range(0, len(image_blobs), self.batch_size):
                batch = image_blobs[i:i + self.batch_size]
                batch_stats = self._process_batch(batch)
                
                # Update statistics
                stats['total_processed'] += batch_stats['total']
                stats['successful'] += batch_stats['successful']
                stats['failed'] += batch_stats['failed']
            
            stats['processing_time'] = time.time() - start_time
            logger.info(f"Processing completed: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing bucket: {str(e)}")
            raise

    def _process_batch(self, blobs: List[storage.Blob]) -> Dict:
        """Process a batch of images in parallel"""
        stats = {'total': len(blobs), 'successful': 0, 'failed': 0}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_blob = {
                executor.submit(self._process_single_image, blob): blob 
                for blob in blobs
            }
            
            for future in concurrent.futures.as_completed(future_to_blob):
                blob = future_to_blob[future]
                try:
                    result = future.result()
                    if result['success']:
                        stats['successful'] += 1
                    else:
                        stats['failed'] += 1
                except Exception as e:
                    logger.error(f"Error processing {blob.name}: {str(e)}")
                    stats['failed'] += 1
        
        return stats

    def _process_single_image(self, blob: storage.Blob) -> Dict:
        """Process a single prescription image"""
        try:
            # Download image
            local_path = os.path.join(self.download_dir, blob.name.replace('/', '_'))
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded: {blob.name}")
            
            # Process image through pipeline
            results = self._run_analysis_pipeline(local_path)
            
            # Save results
            output_path = self._save_results(results, blob.name)
            
            # Cleanup
            os.remove(local_path)
            
            return {
                'success': True,
                'input_path': blob.name,
                'output_path': output_path
            }
            
        except Exception as e:
            logger.error(f"Error processing {blob.name}: {str(e)}")
            return {
                'success': False,
                'input_path': blob.name,
                'error': str(e)
            }

    def _run_analysis_pipeline(self, image_path: str) -> Dict:
        """Run the complete analysis pipeline on an image"""
        try:
            # Step 1: OCR Processing
            ocr_results = process_prescription_image(image_path)
            
            # Step 2: NER Analysis
            if 'text' in ocr_results:
                ner_results = analyze_prescription(ocr_results['text'])
            else:
                raise ValueError("No text extracted from image")
            
            # Step 3: Drug Interaction Check
            interactions = {}
            if 'entities' in ner_results and 'MEDICATION' in ner_results['entities']:
                for med in ner_results['entities']['MEDICATION']:
                    interactions[med] = self.drug_checker.get_drug_events(med)
            
            # Step 4: LLM Analysis
            llm_prompt = create_prescription_prompt(
                ocr_results['text'],
                ner_results['entities'],
                interactions
            )
            
            # Combine results
            results = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'ocr_results': ocr_results,
                'ner_results': ner_results,
                'drug_interactions': interactions,
                'llm_prompt': llm_prompt
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            raise

    def _save_results(self, results: Dict, original_path: str) -> str:
        """Save results to GCP Storage"""
        try:
            # Generate result filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            result_filename = f"results/{base_name}_{timestamp}.json"
            
            # Upload to GCP Storage
            blob = self.output_bucket.blob(result_filename)
            blob.upload_from_string(
                json.dumps(results, indent=2),
                content_type='application/json'
            )
            
            logger.info(f"Saved results to: {result_filename}")
            return result_filename
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup temporary files"""
        try:
            for directory in [self.download_dir, self.results_dir]:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Error removing {file_path}: {str(e)}")
                        
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def process_prescriptions(
    project_id: str,
    input_bucket: str,
    output_bucket: str,
    credentials_path: Optional[str] = None,
    prefix: Optional[str] = None
) -> Dict:
    """
    Process prescriptions from GCP Storage
    
    Args:
        project_id: GCP project ID
        input_bucket: Input bucket name
        output_bucket: Output bucket name
        credentials_path: Path to GCP credentials
        prefix: Optional prefix to filter images
        
    Returns:
        Processing statistics
    """
    handler = GCPPrescriptionHandler(
        project_id=project_id,
        input_bucket=input_bucket,
        output_bucket=output_bucket,
        credentials_path=credentials_path
    )
    
    try:
        stats = handler.process_bucket(prefix)
        return stats
    finally:
        handler.cleanup()