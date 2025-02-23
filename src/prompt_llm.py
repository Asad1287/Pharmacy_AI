import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Tuple, Optional, Dict
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class LLMProcessor:
    """
    A class to process prescription text using the Mistral-7B-Instruct model
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """
        Initialize the LLM processor
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LLM Processor with device: {self.device}")

    def setup_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Set up the model and tokenizer
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Initialized model and tokenizer
        """
        try:
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model from {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise

    def create_prompt(self, ocr_text: str, ner_results: Dict, interaction_results: Dict) -> str:
        """
        Create a structured prompt for the model
        
        Args:
            ocr_text (str): Raw OCR text
            ner_results (Dict): Named entity recognition results
            interaction_results (Dict): Drug interaction results
            
        Returns:
            str: Formatted prompt
        """
        prompt = f"""Analyze the following prescription information and provide a detailed summary:

OCR Extracted Text:
{ocr_text}

Identified Medications and Entities:
{self._format_ner_results(ner_results)}

Drug Interaction Information:
{self._format_interaction_results(interaction_results)}

Please provide:
1. A clear summary of the prescription
2. Important warnings or interactions
3. Key instructions for the patient
4. Any potential concerns that should be discussed with the healthcare provider
"""
        return prompt

    def _format_ner_results(self, ner_results: Dict) -> str:
        """Format NER results for the prompt"""
        formatted = ""
        for entity_type, entities in ner_results.items():
            formatted += f"\n{entity_type}:\n"
            for entity in entities:
                formatted += f"- {entity}\n"
        return formatted

    def _format_interaction_results(self, interaction_results: Dict) -> str:
        """Format interaction results for the prompt"""
        formatted = ""
        for med, data in interaction_results.items():
            formatted += f"\n{med}:\n"
            if 'reactions' in data:
                formatted += "Common reactions:\n"
                for reaction, count in list(data['reactions'].items())[:5]:
                    formatted += f"- {reaction}: {count} reports\n"
        return formatted

    def process_text(
        self, 
        input_text: str, 
        max_length: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Process text through the model
        
        Args:
            input_text (str): Text to process
            max_length (int): Maximum length of generated response
            temperature (float): Sampling temperature
            
        Returns:
            str: Model's response
        """
        try:
            if not self.model or not self.tokenizer:
                logger.info("Model not initialized. Setting up model...")
                self.setup_model()

            logger.info("Tokenizing input text")
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            logger.info("Generating response")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def save_analysis(
        self, 
        input_text: str, 
        response: str, 
        output_dir: str = 'prescriptions_output'
    ) -> str:
        """
        Save the analysis results
        
        Args:
            input_text (str): Original input text
            response (str): Model's response
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved file
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"llm_analysis_{timestamp}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PRESCRIPTION ANALYSIS REPORT (LLM)\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("INPUT TEXT:\n")
                f.write("-" * 50 + "\n")
                f.write(input_text + "\n\n")
                
                f.write("ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                f.write(response + "\n\n")
                
                f.write("\nDISCLAIMER:\n")
                f.write("This analysis is provided for informational purposes only.\n")
                f.write("Always consult with healthcare professionals for medical advice.\n")

            logger.info(f"Analysis saved to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise

def process_prescription_results(
    ocr_text: str,
    ner_results: Dict,
    interaction_results: Dict,
    output_dir: str = 'prescriptions_output'
) -> Tuple[str, str]:
    """
    Process prescription analysis results through LLM
    
    Args:
        ocr_text (str): OCR extracted text
        ner_results (Dict): NER results
        interaction_results (Dict): Drug interaction results
        output_dir (str): Output directory
        
    Returns:
        Tuple[str, str]: LLM response and output file path
    """
    try:
        processor = LLMProcessor()
        
        # Create structured prompt
        prompt = processor.create_prompt(ocr_text, ner_results, interaction_results)
        
        # Process through LLM
        response = processor.process_text(prompt)
        
        # Save results
        output_file = processor.save_analysis(prompt, response, output_dir)
        
        return response, output_file
        
    except Exception as e:
        logger.error(f"Error in prescription processing: {str(e)}")
        raise