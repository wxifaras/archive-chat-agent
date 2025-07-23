"""
Validation service for comparing LLM responses against ground truth data.
Adapted for one-shot Q&A validation without conversation context.
"""

import json
import os
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import openai
import csv

from core.config import settings
from prompts.evaluation_prompts import CORRECTNESS_PROMPT

logger = logging.getLogger(__name__)

class EvaluationResponse(BaseModel):
    """Schema for validation response evaluation"""
    question: str = Field(..., description="The question asked")
    ground_truth: str = Field(..., description="The ground truth answer")
    answer: str = Field(..., description="The generated answer")
    thoughts: str = Field(..., description="Evaluation thoughts on the response")
    stars: int = Field(..., description="Rating (1, 3, or 5 stars)")
    
    @field_validator('stars')
    def validate_stars(cls, v):
        """Ensure stars is one of the allowed values: 1, 3, or 5"""
        if v not in [1, 3, 5]:
            raise ValueError(f'stars must be 1, 3, or 5, got {v}')
        return v


class LLMValidationService:
    """Service for validating LLM responses against ground truth data"""
    
    # Models that require specific temperature settings
    TEMPERATURE_RESTRICTED_MODELS = {
        'o1-preview': 1.0,
        'o1-mini': 1.0,
        'o3': 1.0,
        'o3-mini': 1.0,
        'o3-pro': 1.0,
        # Note: o1 and o3 family models only support temperature=1.0
    }
    
    def __init__(self):
        """Initialize the validation service with Azure OpenAI client"""
        self.client = openai.AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        # Use dedicated evaluation model (defaults to gpt-4.1)
        self.deployment_name = settings.AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME
        logger.info(f"LLMValidationService initialized with evaluation model: {self.deployment_name}")
        logger.info(f"Note: RAG pipeline uses model: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
        
        # Determine temperature based on model and settings
        # First check for models with restricted temperatures
        restricted_temp = self._get_restricted_model_temperature(self.deployment_name)
        if restricted_temp is not None:
            self.temperature = restricted_temp
            logger.info(f"Model {self.deployment_name} requires temperature={self.temperature} (restricted)")
        else:
            # Use configurable temperature from settings
            self.temperature = settings.EVALUATION_TEMPERATURE
            logger.info(f"Using configurable temperature={self.temperature} for model {self.deployment_name}")
    
    def _get_restricted_model_temperature(self, model_name: str) -> Optional[float]:
        """Get the required temperature for models with restrictions.
        
        Returns None if the model has no temperature restrictions.
        """
        model_lower = model_name.lower()
        for restricted_model, required_temp in self.TEMPERATURE_RESTRICTED_MODELS.items():
            if restricted_model in model_lower:
                return required_temp
        return None
    
    async def validate_single_qa(
        self, 
        question: str, 
        ground_truth: str, 
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Validate a single Q&A pair without conversation context.
        
        Args:
            question: The question asked
            ground_truth: The expected answer
            generated_answer: The LLM-generated answer
            
        Returns:
            Evaluation result dictionary
        """
        try:
            context = f"Question: {question}\nExpected Answer: {ground_truth}"
            
            answer_json = json.dumps({
                "Question": question,
                "Answer": generated_answer
            })
            
            formatted_prompt = CORRECTNESS_PROMPT.replace("{{context}}", context).replace("{{answer}}", answer_json)
            
            # Call the evaluation LLM with model-specific temperature
            evaluation_response = await self.client.beta.chat.completions.parse(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an AI evaluator for question-answer pairs."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=self.temperature,
                response_format=EvaluationResponse
            )
            
            evaluation = json.loads(evaluation_response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {
                "question": question,
                "ground_truth": ground_truth,
                "answer": generated_answer,
                "thoughts": f"Error during evaluation: {str(e)}",
                "stars": 0
            }
    
    async def validate_from_csv(
        self, 
        csv_path: str,
        query_function=None,
        max_concurrent: int = 5,
        on_question_complete=None
    ) -> List[Dict[str, Any]]:
        """
        Validate responses against a golden dataset CSV with parallel processing.
        
        Args:
            csv_path: Path to golden dataset CSV
            query_function: Optional function that generates answers for questions.
                          If not provided, expects 'generated_answer' column in CSV.
            max_concurrent: Maximum number of concurrent evaluations (default: 5)
            
        Returns:
            List of evaluation results
        """
        
        async def process_single_qa(row_data: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
            """Process a single Q&A pair asynchronously."""
            # Extract metadata early for error handling
            question_id = row_data.get('question_id', f'Q{index+1}')
            set_id = row_data.get('set_id', 'Unknown')
            
            try:
                logger.info(f"Starting evaluation for {set_id}/{question_id} (row {index + 1})")
                
                # Extract question and ground truth
                question = row_data.get('question', '').strip()
                ground_truth = row_data.get('answer', '') or row_data.get('expected_answer', '')
                ground_truth = ground_truth.strip()
                
                if not question or not ground_truth:
                    logger.warning(f"Skipping {set_id}/{question_id}: missing question or answer/expected_answer")
                    return None
                
                if query_function:
                    logger.info(f"Generating answer for {set_id}/{question_id}: {question[:50]}...")
                    generated_answer = await query_function(question)
                    logger.info(f"Answer generated for {set_id}/{question_id}")
                else:
                    generated_answer = row_data.get('generated_answer', '').strip()
                    if not generated_answer:
                        logger.warning(f"Skipping {set_id}/{question_id}: missing generated_answer")
                        return None
                
                # Validate the answer
                logger.info(f"Evaluating answer for {set_id}/{question_id}")
                evaluation = await self.validate_single_qa(
                    question, 
                    ground_truth, 
                    generated_answer
                )
                logger.info(f"Completed {set_id}/{question_id} with {evaluation.get('stars', 0)} stars")
                
                evaluation['metadata'] = {
                    'set_id': row_data.get('set_id', ''),
                    'question_id': row_data.get('question_id', ''),
                    'content_type': row_data.get('content_type', ''),
                    'source_files': row_data.get('source_files', ''),
                    'validation_notes': row_data.get('validation_notes', '')
                }
                
                if on_question_complete:
                    await on_question_complete(question_id, set_id, 'completed')
                
                return evaluation
                
            except Exception as e:
                logger.error(f"Error processing row {index + 1}: {str(e)}")
                if on_question_complete:
                    await on_question_complete(question_id, set_id, 'failed')
                return None
        
        try:
            rows = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            logger.info(f"Loaded {len(rows)} rows from CSV")
            
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(row, index):
                async with semaphore:
                    # Add a small delay to avoid overwhelming the API
                    await asyncio.sleep(0.1 * (index % max_concurrent))
                    return await process_single_qa(row, index)
            
            # Process all rows concurrently with limited parallelism
            tasks = [
                process_with_semaphore(row, i) 
                for i, row in enumerate(rows)
            ]
            
            logger.info(f"Starting parallel evaluation with max {max_concurrent} concurrent tasks")
            
            completed = 0
            results_with_none = []
            
            batch_size = max_concurrent
            for batch_start in range(0, len(tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size} (questions {batch_start + 1}-{batch_end}/{len(tasks)})")
                
                batch_results = await asyncio.gather(*batch_tasks)
                results_with_none.extend(batch_results)
                
                completed = batch_end
                progress_pct = (completed / len(tasks)) * 100
                logger.info(f"Progress: {completed}/{len(tasks)} questions processed ({progress_pct:.1f}%)")
            
            results = [r for r in results_with_none if r is not None]
            logger.info(f"Evaluation complete: {len(results)} valid results from {len(rows)} rows")
            
            # Write results to a JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join("tests", "validation_results", f"validation_results_{timestamp}.json")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Validation complete. Results written to {results_file}")
            
            # Calculate summary statistics
            total = len(results)
            if total > 0:
                five_stars = sum(1 for r in results if r.get('stars') == 5)
                three_stars = sum(1 for r in results if r.get('stars') == 3)
                one_star = sum(1 for r in results if r.get('stars') == 1)
                
                summary = {
                    "total_evaluations": total,
                    "five_star_count": five_stars,
                    "three_star_count": three_stars,
                    "one_star_count": one_star,
                    "five_star_percentage": (five_stars / total) * 100,
                    "three_star_percentage": (three_stars / total) * 100,
                    "one_star_percentage": (one_star / total) * 100,
                    "results_file": results_file
                }
                
                return {"results": results, "summary": summary}
            
            return {"results": results, "summary": {"total_evaluations": 0}}
            
        except Exception as e:
            logger.error(f"Error in CSV validation: {str(e)}")
            raise
    
