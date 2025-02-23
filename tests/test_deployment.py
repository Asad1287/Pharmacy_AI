import requests
import os
from google.cloud import storage
import time

def test_deployment():
    """Test the deployed prescription analyzer"""
    
    # Configuration
    project_id = os.getenv('PROJECT_ID')
    service_url = os.getenv('SERVICE_URL')
    test_image = "test_prescription.jpg"
    
    # Initialize GCS client
    storage_client = storage.Client()
    input_bucket = storage_client.bucket(f"{project_id}-prescription-images")
    output_bucket = storage_client.bucket(f"{project_id}-prescription-results")
    
    try:
        # Upload test image
        print("Uploading test image...")
        blob = input_bucket.blob(f"test/{test_image}")
        blob.upload_from_filename(test_image)
        
        # Wait for processing
        print("Waiting for processing...")
        time.sleep(30)
        
        # Check results
        print("Checking results...")
        results = list(output_bucket.list_blobs(prefix="results/test"))
        
        if results:
            print("Test successful! Results found in output bucket.")
            for result in results:
                print(f"Result file: {result.name}")
        else:
            print("No results found in output bucket.")
            
    except Exception as e:
        print(f"Error testing deployment: {str(e)}")
        
if __name__ == "__main__":
    test_deployment()