import pytesseract
import easyocr
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Dict, List, Union, Tuple, Optional
import time
import logging
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    A class to handle OCR processing using both EasyOCR and Tesseract
    with GPU acceleration support
    """
    
    def __init__(self, lang: str = 'en', use_gpu: bool = None):
        """
        Initialize the OCR processor
        
        Args:
            lang: Language code for OCR
            use_gpu: Whether to use GPU. If None, automatically detect
        """
        self.lang = lang
        self.device = self._setup_device(use_gpu)
        self.reader = self._initialize_easyocr()
        self.transform = self._setup_transforms()
        logger.info(f"OCR Processor initialized using {self.device}")

    def _setup_device(self, use_gpu: Optional[bool]) -> torch.device:
        """Setup processing device"""
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        return device

    def _initialize_easyocr(self) -> easyocr.Reader:
        """Initialize EasyOCR reader"""
        try:
            return easyocr.Reader(
                [self.lang],
                gpu=self.device.type == 'cuda',
                model_storage_directory='models'
            )
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {str(e)}")
            raise

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transformations"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced image enhancement techniques
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Create CLAHE object for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # GPU-accelerated denoising if available
            if self.device.type == 'cuda':
                img_tensor = torch.from_numpy(enhanced).float().unsqueeze(0).unsqueeze(0)
                img_tensor = img_tensor.to(self.device)

                with torch.no_grad():
                    # Bilateral filtering simulation
                    gaussian_kernel = torch.tensor([
                        [1, 4, 6, 4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1, 4, 6, 4, 1]
                    ], dtype=torch.float32, device=self.device) / 256

                    gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5)
                    img_tensor = torch.nn.functional.conv2d(
                        img_tensor,
                        gaussian_kernel,
                        padding=2
                    )

                    denoised = img_tensor.squeeze().cpu().numpy()
            else:
                denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)

            # Convert back to uint8
            denoised = (denoised * 255).astype(np.uint8) if denoised.dtype != np.uint8 else denoised

            return denoised

        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return gray

    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using GPU acceleration if available
        
        Args:
            image: Input image array
            
        Returns:
            Deskewed image array
        """
        try:
            # Convert to binary
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Find all non-zero points
            coords = np.column_stack(np.where(thresh > 0))

            if len(coords) < 20:
                return image

            if self.device.type == 'cuda':
                # GPU accelerated angle calculation
                coords_tensor = torch.from_numpy(coords).float().to(self.device)
                
                with torch.no_grad():
                    mean = torch.mean(coords_tensor, dim=0)
                    centered_coords = coords_tensor - mean
                    cov = torch.mm(centered_coords.t(), centered_coords)
                    _, eigenvectors = torch.linalg.eigh(cov)
                    angle = torch.atan2(eigenvectors[-1, 1], eigenvectors[-1, 0])
                    angle = torch.rad2deg(angle).cpu().numpy()
            else:
                angle = cv2.minAreaRect(coords)[-1]

            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = -90 + angle

            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            return rotated

        except Exception as e:
            logger.error(f"Error deskewing image: {str(e)}")
            return image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        try:
            # Initial enhancement
            enhanced = self.enhance_image(image)

            # Deskew
            deskewed = self.deskew_image(enhanced)

            # Additional preprocessing
            img_tensor = self.transform(Image.fromarray(deskewed)).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                # Additional denoising
                img_tensor = torch.nn.functional.avg_pool2d(
                    img_tensor, 2, stride=1, padding=1
                )

                # Enhance edges
                kernel = torch.tensor([
                    [-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]
                ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
                img_tensor = torch.nn.functional.conv2d(
                    img_tensor, kernel, padding=1
                )

                # Normalize
                img_tensor = torch.nn.functional.normalize(img_tensor, dim=1)

            # Convert back to numpy
            processed = (img_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

            # Final adjustments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            processed = cv2.dilate(processed, kernel, iterations=1)
            processed = cv2.medianBlur(processed, 3)

            return processed

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return self.enhance_image(image)

    def process_image(
        self,
        image_path: str,
        save_dir: str = 'prescriptions_output',
        use_tesseract: bool = True
    ) -> Dict:
        """
        Process an image using both EasyOCR and Tesseract
        
        Args:
            image_path: Path to the image file
            save_dir: Directory to save results
            use_tesseract: Whether to also use Tesseract
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            start_time = time.time()

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")

            # Preprocess
            processed = self.preprocess_image(image)

            # EasyOCR processing
            easyocr_results = self.reader.readtext(processed)
            
            # Format EasyOCR results
            results = {
                'easyocr': [
                    {
                        'text': text,
                        'confidence': round(float(conf) * 100, 2),
                        'bbox': bbox
                    }
                    for bbox, text, conf in easyocr_results
                ]
            }

            # Tesseract processing if requested
            if use_tesseract:
                custom_config = r'--oem 3 --psm 6'
                tesseract_text = pytesseract.image_to_string(
                    processed,
                    lang='eng',
                    config=custom_config
                )
                
                # Get confidence scores
                tesseract_data = pytesseract.image_to_data(
                    processed,
                    lang='eng',
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                
                conf_scores = [int(conf) for conf in tesseract_data['conf'] if conf != '-1']
                avg_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 0

                results['tesseract'] = {
                    'text': tesseract_text.strip(),
                    'confidence': round(avg_confidence, 2)
                }

            # Calculate processing time
            process_time = time.time() - start_time
            results['process_time'] = round(process_time, 3)

            # Save results
            self.save_results(results, image_path, save_dir)

            return results

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {
                'error': str(e),
                'process_time': 0
            }

    def save_results(
        self,
        results: Dict,
        image_path: str,
        save_dir: str = 'prescriptions_output'
    ) -> str:
        """
        Save OCR results to file
        
        Args:
            results: OCR results dictionary
            image_path: Original image path
            save_dir: Output directory
            
        Returns:
            Path to saved file
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = os.path.join(save_dir, f"{base_name}_{timestamp}_ocr.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("OCR ANALYSIS RESULTS\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Image: {image_path}\n")
                f.write(f"Processing Time: {results['process_time']} seconds\n")
                f.write("-" * 80 + "\n\n")

                # EasyOCR results
                f.write("EASYOCR RESULTS:\n")
                f.write("-" * 50 + "\n")
                for item in results['easyocr']:
                    f.write(f"Text: {item['text']}\n")
                    f.write(f"Confidence: {item['confidence']}%\n")
                    f.write(f"Position: {item['bbox']}\n")
                    f.write("-" * 30 + "\n")

                # Tesseract results if available
                if 'tesseract' in results:
                    f.write("\nTESSERACT RESULTS:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Text:\n{results['tesseract']['text']}\n\n")
                    f.write(f"Confidence: {results['tesseract']['confidence']}%\n")

            logger.info(f"Results saved to: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def process_prescription_image(
    image_path: str,
    save_dir: str = 'prescriptions_output',
    use_gpu: bool = None
) -> Dict:
    """
    Process a prescription image using OCR
    
    Args:
        image_path: Path to the image file
        save_dir: Directory to save results
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary containing OCR results
    """
    processor = OCRProcessor(use_gpu=use_gpu)
    return processor.process_image(image_path, save_dir)