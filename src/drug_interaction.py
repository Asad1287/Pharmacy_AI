import requests
from datetime import datetime
from typing import List, Dict, Optional, Union
import logging
import os
import json

# Configure logging
logger = logging.getLogger(__name__)

class DrugInteractionChecker:
    """
    A class to check drug interactions and adverse events using the OpenFDA API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the drug interaction checker
        
        Args:
            api_key (str): OpenFDA API key (required)
        """
        if not api_key:
            raise ValueError("OpenFDA API key is required")
            
        self.api_key = api_key
        self.base_url = "https://api.fda.gov/drug/event.json"
        logger.info("DrugInteractionChecker initialized")

    def extract_medications_from_file(self, filename: str) -> List[str]:
        """
        Extract medication names from the analysis results file

        Args:
            filename (str): Path to the analysis results file

        Returns:
            List[str]: List of medication names found in the file
        """
        medications = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all medication sections
            sections = content.split('PRESCRIPTION #')
            for section in sections:
                if 'MEDICATION:' in section:
                    med_section = section.split('MEDICATION:')[1].split('\n')[1]
                    medication = med_section.strip('• ').strip()
                    if medication:
                        medications.append(medication)

            logger.info(f"Extracted {len(medications)} medications from file")
            return list(set(medications))  # Remove duplicates

        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return []
        except Exception as e:
            logger.error(f"Error reading file {filename}: {str(e)}")
            return []

    def get_drug_events(self, drug_name: str) -> Dict:
        """
        Get drug event information from OpenFDA

        Args:
            drug_name (str): Name of the drug to check

        Returns:
            Dict: Dictionary containing drug event information or error message
        """
        try:
            logger.info(f"Fetching drug events for: {drug_name}")
            params = {
                'search': f'patient.drug.medicinalproduct:{drug_name}',
                'limit': 100,
                'api_key': self.api_key
            }

            response = requests.get(
                self.base_url,
                params=params,
                timeout=10  # Add timeout for the request
            )
            response.raise_for_status()
            data = response.json()

            if 'results' not in data:
                logger.warning(f"No results found for drug: {drug_name}")
                return {'error': 'No results found'}

            # Process and summarize the events
            events_summary = self._process_events(data['results'])
            logger.info(f"Successfully processed events for {drug_name}")
            return events_summary

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg = f"{error_msg}\nResponse: {e.response.text}"
            logger.error(f"API request failed for {drug_name}: {error_msg}")
            return {'error': f'API request failed: {error_msg}'}
        except Exception as e:
            logger.error(f"Unexpected error processing {drug_name}: {str(e)}")
            return {'error': f'Processing error: {str(e)}'}

    def _process_events(self, events: List[Dict]) -> Dict:
        """
        Process and summarize drug events

        Args:
            events (List[Dict]): List of drug events from OpenFDA

        Returns:
            Dict: Summarized event information
        """
        summary = {
            'total_reports': len(events),
            'reactions': {},
            'interactions': [],
            'common_indications': set(),
            'warnings': set()
        }

        try:
            for event in events:
                if 'patient' in event:
                    patient = event['patient']

                    # Process reactions
                    if 'reaction' in patient:
                        for reaction in patient['reaction']:
                            if 'reactionmeddrapt' in reaction:
                                reaction_term = reaction['reactionmeddrapt']
                                summary['reactions'][reaction_term] = \
                                    summary['reactions'].get(reaction_term, 0) + 1

                    # Process drug information
                    if 'drug' in patient:
                        for drug in patient['drug']:
                            # Get indications
                            if 'drugindication' in drug:
                                summary['common_indications'].add(drug['drugindication'])

                            # Look for interactions
                            if drug.get('drugcharacterization') == '2':  # Concomitant medications
                                if 'medicinalproduct' in drug:
                                    summary['interactions'].append(drug['medicinalproduct'])

            # Convert sets to lists for JSON serialization
            summary['common_indications'] = list(summary['common_indications'])
            summary['warnings'] = list(summary['warnings'])

            # Sort reactions by frequency
            summary['reactions'] = dict(sorted(
                summary['reactions'].items(),
                key=lambda x: x[1],
                reverse=True
            ))

            return summary

        except Exception as e:
            logger.error(f"Error processing events: {str(e)}")
            return {
                'total_reports': 0,
                'reactions': {},
                'interactions': [],
                'common_indications': [],
                'warnings': [f"Error processing events: {str(e)}"]
            }

    def save_interaction_results(self, 
                               results: Dict, 
                               filename: str = 'drug_safety_report.txt') -> None:
        """
        Save drug safety and interaction results to a file

        Args:
            results (Dict): Dictionary of drug interaction results
            filename (str): Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DRUG SAFETY AND INTERACTION REPORT\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Medications Analyzed: {len(results)}\n\n")

                for med_name, data in results.items():
                    f.write("-" * 80 + "\n")
                    f.write(f"MEDICATION: {med_name}\n")
                    f.write("-" * 80 + "\n\n")

                    if 'error' in data:
                        f.write(f"Error: {data['error']}\n\n")
                        continue

                    # Total reports
                    f.write(f"Total Reports Analyzed: {data['total_reports']}\n\n")

                    # Common reactions
                    f.write("TOP REPORTED REACTIONS:\n")
                    for reaction, count in list(data['reactions'].items())[:10]:
                        f.write(f"• {reaction}: {count} reports\n")
                    f.write("\n")

                    # Drug interactions
                    if data['interactions']:
                        f.write("REPORTED CONCURRENT MEDICATIONS:\n")
                        for interaction in set(data['interactions']):
                            f.write(f"• {interaction}\n")
                        f.write("\n")

                    # Common indications
                    if data['common_indications']:
                        f.write("COMMON INDICATIONS:\n")
                        for indication in data['common_indications']:
                            f.write(f"• {indication}\n")
                        f.write("\n")

                f.write("\nDISCLAIMER:\n")
                f.write("This report is based on FDA adverse event reporting data and should not be used\n")
                f.write("to make medical decisions. Consult healthcare professionals for medical advice.\n")

            logger.info(f"Drug safety report saved to: {filename}")

        except Exception as e:
            logger.error(f"Error saving results to {filename}: {str(e)}")
            raise

def test_api_key(api_key: str) -> bool:
    """
    Test if the provided OpenFDA API key is valid

    Args:
        api_key (str): OpenFDA API key to test

    Returns:
        bool: True if the API key is valid, False otherwise
    """
    try:
        test_url = "https://api.fda.gov/drug/event.json"
        params = {
            'limit': 1,
            'api_key': api_key
        }
        response = requests.get(test_url, params=params, timeout=5)
        return response.status_code == 200
    except:
        return False