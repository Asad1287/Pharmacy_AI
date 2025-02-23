import spacy
from spacy.tokens import Doc
import requests
from typing import List, Dict, Set, Union, Optional
import logging
import re
from datetime import datetime
import os
import json

# Configure logging
logger = logging.getLogger(__name__)

class MedicationNER:
    """
    Named Entity Recognition system for medication-related information
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the NER system with predefined patterns and rules
        
        Args:
            api_key: OpenFDA API key for medication validation
        """
        self.api_key = api_key
        self.nlp = spacy.blank("en")
        self.medications_cache = set()
        self.last_cache_update = None
        
        # Define entity patterns
        self.entity_types = {
            'MEDICATION': None,  # Will be populated from OpenFDA
            'DOSAGE': [
                r'\d+\s*(?:mg|mL|tablet|tablets|puffs|grams|drops|capsule|cap)',
                r'\d+\s*(?:gram|g|mcg)',
                r'one|two|three|four|five|six|seven|eight|nine|ten\s+tablets?'
            ],
            'STRENGTH': [
                r'\d+\s*(?:mg|mcg|%)',
                r'\d+\.\d+\s*(?:mg|mcg|%)',
                r'\d+/\d+\s*(?:mg|mcg|%)'
            ],
            'ROUTE': [
                'orally', 'by mouth', 'topically', 'intramuscularly',
                'subcutaneously', 'inhale', 'via inhaler', 'into each eye',
                'under the tongue', 'sublingual', 'intravenously'
            ],
            'FREQUENCY': [
                'every 8 hours', 'twice daily', 'three times a day',
                'once daily', 'daily', 'every morning', 'every evening',
                'single dose', 'every 12 hours', 'every 4 to 6 hours',
                'as needed', 'before meals', 'after meals', 'at bedtime'
            ],
            'DURATION': [
                r'\d+\s*(?:days|weeks|months)',
                r'\d+\s*(?:day|week|month)',
                r'for \d+ days',
                'until finished'
            ],
            'FORM': [
                'cream', 'ointment', 'gel', 'lotion', 'solution', 'syrup',
                'suspension', 'tablet', 'capsule', 'suppository', 'inhaler',
                'drops', 'patch', 'injection'
            ],
            'TIME': [
                'morning', 'afternoon', 'evening', 'night', 'bedtime',
                'before meals', 'after meals', 'with meals'
            ],
            'CONDITION': [
                'as needed for pain', 'for fever', 'when required',
                'for blood pressure', 'for diabetes', 'for infection'
            ]
        }
        
        # Initialize medication list
        if self.api_key:
            self._update_medications_cache()

    def _update_medications_cache(self) -> None:
        """Update the medications cache from OpenFDA API"""
        try:
            if (not self.last_cache_update or 
                (datetime.now() - self.last_cache_update).days >= 1):
                
                medications = self.fetch_medications_from_openfda()
                if medications:
                    self.medications_cache = set(medications)
                    self.entity_types['MEDICATION'] = list(self.medications_cache)
                    self.last_cache_update = datetime.now()
                    logger.info("Successfully updated medications cache")
                
        except Exception as e:
            logger.error(f"Error updating medications cache: {str(e)}")

    def fetch_medications_from_openfda(self) -> List[str]:
        """
        Fetch medication names from OpenFDA API
        
        Returns:
            List of medication names
        """
        try:
            medications = set()
            base_url = "https://api.fda.gov/drug/ndc.json"
            limit = 1000
            total_requests = 5

            for skip in range(0, limit * total_requests, limit):
                params = {
                    'search': 'product_type:"HUMAN PRESCRIPTION DRUG"',
                    'limit': limit,
                    'skip': skip,
                    'api_key': self.api_key
                }

                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'results' in data:
                    for result in data['results']:
                        # Add generic names
                        if 'generic_name' in result:
                            generic_names = result['generic_name'].split(', ')
                            medications.update(generic_names)

                        # Add brand names
                        if 'brand_name' in result:
                            brand_names = result['brand_name'].split(', ')
                            medications.update(brand_names)

                        # Add active ingredients
                        if 'active_ingredients' in result:
                            for ingredient in result['active_ingredients']:
                                if 'name' in ingredient:
                                    medications.add(ingredient['name'])

            # Clean and standardize medication names
            cleaned_medications = set()
            for med in medications:
                cleaned_med = med.title().strip()
                if len(cleaned_med) > 2 and cleaned_med.replace(' ', '').isalnum():
                    cleaned_medications.add(cleaned_med)

            return list(cleaned_medications)

        except Exception as e:
            logger.error(f"Error fetching medications from OpenFDA: {str(e)}")
            return []

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medication-related entities from text
        
        Args:
            text: Input prescription text
            
        Returns:
            Dictionary of entity types and their values
        """
        try:
            # Update medications cache if needed
            if self.api_key:
                self._update_medications_cache()

            entities = {entity_type: [] for entity_type in self.entity_types.keys()}
            doc = self.nlp(text)

            # Process each entity type
            for entity_type, patterns in self.entity_types.items():
                if patterns:  # For pattern-based entities
                    for pattern in patterns:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            entity = match.group().strip()
                            if entity not in entities[entity_type]:
                                entities[entity_type].append(entity)
                else:  # For medication names from cache
                    words = text.split()
                    for i in range(len(words)):
                        for j in range(i + 1, len(words) + 1):
                            phrase = ' '.join(words[i:j]).strip().title()
                            if phrase in self.medications_cache:
                                if phrase not in entities[entity_type]:
                                    entities[entity_type].append(phrase)

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {entity_type: [] for entity_type in self.entity_types.keys()}

    def format_entities(self, entities: Dict[str, List[str]]) -> str:
        """
        Format extracted entities into a readable string
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Formatted string representation
        """
        output = []
        
        for entity_type, values in entities.items():
            if values:
                output.append(f"\n{entity_type}:")
                for value in values:
                    output.append(f"  • {value}")
        
        return '\n'.join(output)

    def process_prescription(
        self,
        text: str,
        save_results: bool = True,
        output_dir: str = 'prescriptions_output'
    ) -> Dict:
        """
        Process prescription text and extract all relevant information
        
        Args:
            text: Prescription text
            save_results: Whether to save results to file
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            # Extract entities
            entities = self.extract_entities(text)
            
            # Prepare results
            results = {
                'original_text': text,
                'entities': entities,
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'medication_count': len(entities.get('MEDICATION', [])),
                    'has_dosage': len(entities.get('DOSAGE', [])) > 0,
                    'has_frequency': len(entities.get('FREQUENCY', [])) > 0,
                    'has_duration': len(entities.get('DURATION', [])) > 0
                }
            }

            # Save results if requested
            if save_results:
                self.save_results(results, output_dir)

            return results

        except Exception as e:
            logger.error(f"Error processing prescription: {str(e)}")
            return {
                'error': str(e),
                'original_text': text
            }

    def save_results(
        self,
        results: Dict,
        output_dir: str = 'prescriptions_output'
    ) -> str:
        """
        Save analysis results to file
        
        Args:
            results: Analysis results
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"ner_analysis_{timestamp}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PRESCRIPTION NER ANALYSIS\n")
                f.write("=" * 80 + "\n\n")

                f.write("ORIGINAL TEXT:\n")
                f.write("-" * 50 + "\n")
                f.write(results['original_text'] + "\n\n")

                f.write("EXTRACTED ENTITIES:\n")
                f.write("-" * 50 + "\n")
                f.write(self.format_entities(results['entities']) + "\n\n")

                f.write("ANALYSIS SUMMARY:\n")
                f.write("-" * 50 + "\n")
                for key, value in results['analysis'].items():
                    f.write(f"{key}: {value}\n")

            logger.info(f"Results saved to: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

def analyze_prescription(
    text: str,
    api_key: Optional[str] = None,
    save_results: bool = True
) -> Dict:
    """
    Analyze prescription text using NER
    
    Args:
        text: Prescription text to analyze
        api_key: OpenFDA API key
        save_results: Whether to save results
        
    Returns:
        Dictionary containing analysis results
    """
    ner = MedicationNER(api_key=api_key)
    return ner.process_prescription(text, save_results)