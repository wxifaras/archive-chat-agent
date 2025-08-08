"""
Validation service for comparing LLM responses against ground truth data.
Adapted for one-shot Q&A validation without conversation context.
"""

import json
import os
import re
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import openai
import csv
from pathlib import Path

from core.config import settings
from prompts.evaluation_prompts import CORRECTNESS_PROMPT
from prompts.conversation_evaluation_prompts import (
    CONVERSATION_TURN_EVALUATION_PROMPT,
    CONVERSATION_HOLISTIC_EVALUATION_PROMPT,
    CONTEXT_DEPENDENT_EVALUATION_PROMPT
)
from models.conversation_evaluation import (
    ConversationTurn,
    ConversationEvaluationRequest,
    ConversationEvaluationResult,
    TurnEvaluation,
    ConversationMetrics,
    ConversationRole,
    EvaluationMode
)
from utils.conversation_csv_parser import ConversationCSVParser

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
        
        # ContentService will be initialized when needed for conversation evaluations
        self.content_service = None
        
        # Store session mappings for conversations
        self.conversation_sessions: Dict[str, str] = {}
    
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
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompts."""
        if not history:
            return "No previous conversation."
        
        formatted_turns = []
        for turn in history:
            role = turn['role'].capitalize()
            message = turn['message']
            formatted_turns.append(f"{role}: {message}")
        
        return "\n".join(formatted_turns)
    
    async def _evaluate_conversation_turn(
        self,
        turn: ConversationTurn,
        generated_answer: str,
        conversation_history: List[Dict[str, str]]
    ) -> TurnEvaluation:
        """Evaluate a single conversation turn with context."""
        try:
            # Format conversation history
            history_text = self._format_conversation_history(conversation_history)
            
            # Format the evaluation prompt
            prompt = CONVERSATION_TURN_EVALUATION_PROMPT.format(
                conversation_history=history_text,
                current_question=turn.message,
                expected_answer=turn.expected_response or "N/A",
                generated_answer=generated_answer
            )
            
            # Call evaluation model
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI conversation evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            eval_result = json.loads(response.choices[0].message.content)
            
            # Create TurnEvaluation object
            return TurnEvaluation(
                turn_number=turn.turn_number,
                question=turn.message,
                expected_response=turn.expected_response or "",
                generated_response=generated_answer,
                rating=eval_result.get('rating', 1),
                correctness_assessment=eval_result.get('correctness_assessment', ''),
                context_usage_assessment=eval_result.get('context_usage_assessment', ''),
                evaluation_thoughts=eval_result.get('evaluation_thoughts', '')
            )
            
        except Exception as e:
            logger.error(f"Error evaluating turn {turn.turn_number}: {str(e)}")
            return TurnEvaluation(
                turn_number=turn.turn_number,
                question=turn.message,
                expected_response=turn.expected_response or "",
                generated_response=generated_answer,
                rating=1,
                correctness_assessment="Error during evaluation",
                context_usage_assessment="Error during evaluation",
                evaluation_thoughts=f"Evaluation error: {str(e)}"
            )
    
    async def _evaluate_conversation_overall(
        self,
        conversation_id: str,
        session_id: str,
        conversation_history: List[Dict[str, str]],
        turn_evaluations: List[TurnEvaluation]
    ) -> Tuple[ConversationMetrics, Dict[str, Any]]:
        """Evaluate the overall conversation quality."""
        try:
            # Format full conversation for evaluation
            full_conversation = self._format_conversation_history(conversation_history)
            
            # Create evaluation prompt
            prompt = CONVERSATION_HOLISTIC_EVALUATION_PROMPT.format(
                full_conversation=full_conversation
            )
            
            # Call evaluation model
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI conversation evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            eval_result = json.loads(response.choices[0].message.content)
            
            # Create ConversationMetrics
            metrics = ConversationMetrics(
                context_coherence_score=eval_result.get('context_coherence_score', 0.5),
                follow_up_accuracy=eval_result.get('follow_up_accuracy', 0.5),
                information_consistency=eval_result.get('information_consistency', 0.5),
                conversation_flow_score=eval_result.get('conversation_flow_score', 0.5),
                overall_effectiveness=eval_result.get('overall_effectiveness', 0.5)
            )
            
            # Extract additional info
            additional_info = {
                'overall_rating': eval_result.get('overall_rating', 3),
                'overall_evaluation_thoughts': eval_result.get('overall_evaluation_thoughts', ''),
                'strengths': eval_result.get('strengths', []),
                'weaknesses': eval_result.get('weaknesses', []),
                'specific_examples': eval_result.get('specific_examples', {})
            }
            
            return metrics, additional_info
            
        except Exception as e:
            logger.error(f"Error in overall conversation evaluation: {str(e)}")
            # Return default metrics on error
            default_metrics = ConversationMetrics(
                context_coherence_score=0.5,
                follow_up_accuracy=0.5,
                information_consistency=0.5,
                conversation_flow_score=0.5,
                overall_effectiveness=0.5
            )
            default_info = {
                'overall_rating': 3,
                'overall_evaluation_thoughts': f"Error during evaluation: {str(e)}",
                'strengths': [],
                'weaknesses': ['Evaluation error occurred']
            }
            return default_metrics, default_info
    
    async def validate_conversation(
        self,
        conversation_csv_path: str,
        evaluation_mode: EvaluationMode = EvaluationMode.CONTEXTUAL,
        max_concurrent: int = 1  # Sequential for conversations to maintain context
    ) -> List[ConversationEvaluationResult]:
        """
        Validate conversations from CSV using the existing RAG pipeline.
        
        Args:
            conversation_csv_path: Path to CSV file with conversations
            evaluation_mode: How to evaluate (contextual, turn_by_turn, holistic)
            max_concurrent: Number of concurrent conversations (default 1)
            
        Returns:
            List of conversation evaluation results
        """
        try:
            # Parse conversations from CSV
            conversations = ConversationCSVParser.parse_csv_file(conversation_csv_path)
            results = []
            
            # Process each conversation
            for conv_id, turns in conversations.items():
                logger.info(f"Evaluating conversation {conv_id} with {len(turns)} turns")
                
                # Generate unique session_id for this conversation
                session_id = f"eval_conv_{conv_id}_{uuid.uuid4()}"
                self.conversation_sessions[conv_id] = session_id
                
                user_id = "evaluation_bot"
                conversation_history = []
                turn_evaluations = []
                
                # Process each turn in the conversation
                for turn in turns:
                    if turn.role == ConversationRole.USER and turn.expected_response:
                        # Generate response using RAG pipeline with same session_id
                        logger.info(f"Generating response for {conv_id} turn {turn.turn_number}")
                        
                        try:
                            # Initialize ContentService if not already done
                            if self.content_service is None:
                                from services.content_service import ContentService
                                self.content_service = ContentService()
                            
                            # Call RAG pipeline - it automatically uses chat history
                            response = await self.content_service.chat_with_content(
                                message=turn.message,
                                user_id=user_id,
                                session_id=session_id  # Same session = automatic context
                            )
                            
                            generated_answer = response.get('answer', '')
                            
                            # Evaluate this turn with context
                            if evaluation_mode != EvaluationMode.HOLISTIC:
                                turn_eval = await self._evaluate_conversation_turn(
                                    turn=turn,
                                    generated_answer=generated_answer,
                                    conversation_history=conversation_history
                                )
                                turn_evaluations.append(turn_eval)
                            
                            # Update conversation history
                            conversation_history.append({
                                "role": "user",
                                "message": turn.message
                            })
                            conversation_history.append({
                                "role": "assistant",
                                "message": generated_answer
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing turn {turn.turn_number} in {conv_id}: {str(e)}")
                            # Add error turn evaluation
                            turn_evaluations.append(TurnEvaluation(
                                turn_number=turn.turn_number,
                                question=turn.message,
                                expected_response=turn.expected_response or "",
                                generated_response=f"Error: {str(e)}",
                                rating=1,
                                correctness_assessment="Error generating response",
                                context_usage_assessment="N/A",
                                evaluation_thoughts=f"Error: {str(e)}"
                            ))
                
                # Evaluate overall conversation
                metrics, additional_info = await self._evaluate_conversation_overall(
                    conversation_id=conv_id,
                    session_id=session_id,
                    conversation_history=conversation_history,
                    turn_evaluations=turn_evaluations
                )
                
                # Create result
                result = ConversationEvaluationResult(
                    conversation_id=conv_id,
                    session_id=session_id,
                    evaluation_mode=evaluation_mode,
                    overall_rating=additional_info['overall_rating'],
                    turn_evaluations=turn_evaluations,
                    conversation_metrics=metrics,
                    overall_evaluation_thoughts=additional_info['overall_evaluation_thoughts'],
                    strengths=additional_info.get('strengths', []),
                    weaknesses=additional_info.get('weaknesses', [])
                )
                
                results.append(result)
                logger.info(f"Completed evaluation of conversation {conv_id}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path("tests/validation_results") / f"conversation_validation_{timestamp}.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert results to dict for JSON serialization
            results_dict = [r.model_dump(mode='json') for r in results]
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Conversation validation complete. Results written to {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in conversation validation: {str(e)}")
            raise
