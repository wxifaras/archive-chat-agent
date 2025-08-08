"""
Evaluation runner service that handles background evaluation tasks with proper isolation.
"""

import asyncio
import logging
import uuid
import os
from typing import Callable, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from models.evaluation_progress import (
    EvaluationProgress, 
    QuestionProgress, 
    QuestionStatus,
    ConversationEvaluationProgress,
    ConversationProgress,
    ConversationStatus,
    ConversationTurnProgress
)
from models.conversation_evaluation import EvaluationMode

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Handles running evaluations in isolated background tasks."""
    
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_progress: Dict[str, EvaluationProgress] = {}
        self.conversation_progress: Dict[str, ConversationEvaluationProgress] = {}
    
    async def run_golden_dataset_evaluation(self, golden_dataset_path: Path) -> str:
        """
        Run golden dataset evaluation in an isolated task.
        Returns task ID for tracking.
        """
        task_id = str(uuid.uuid4())
        
        # Create the task with proper isolation
        task = asyncio.create_task(
            self._run_evaluation_isolated(golden_dataset_path, task_id)
        )
        
        # Store task for tracking
        self.active_tasks[task_id] = task
        
        task.add_done_callback(lambda t: self._cleanup_task(task_id, t))
        
        return task_id
    
    async def _run_evaluation_isolated(self, golden_dataset_path: Path, task_id: str):
        """Run evaluation in an isolated context with its own service instances."""
        try:
            logger.info(f"Starting evaluation task {task_id}")
            
            # Import and create fresh service instances for this task
            # Import here to ensure proper initialization in the task context
            from .llm_validation_service import LLMValidationService
            from .content_service import ContentService
            from core.config import settings
            import csv
            
            progress = EvaluationProgress(task_id=task_id, total_questions=0)
            
            # Read the CSV to get question details
            questions = []
            with open(golden_dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('question'):
                        question_id = row.get('question_id', '')
                        set_id = row.get('set_id', '')
                        question_text = row.get('question', '')
                        
                        q_progress = QuestionProgress(
                            question_id=question_id,
                            set_id=set_id,
                            question=question_text
                        )
                        progress.questions[f"{set_id}_{question_id}"] = q_progress
                        questions.append((question_id, set_id, question_text))
            
            progress.total_questions = len(questions)
            self.task_progress[task_id] = progress
            logger.info(f"Task {task_id}: Found {progress.total_questions} questions to evaluate")
            
            validation_service = LLMValidationService()
            content_service = ContentService()
            
            async def query_function(question: str, question_id: str = None, set_id: str = None) -> str:
                try:
                    # Update progress - starting to generate answer
                    if question_id and set_id:
                        key = f"{set_id}_{question_id}"
                        if key in progress.questions:
                            progress.questions[key].start()
                            logger.info(f"Task {task_id}: Starting Q{question_id} in {set_id} - {question[:50]}...")
                    
                    eval_session_id = f"eval_golden_{uuid.uuid4()}"
                    eval_user_id = "evaluation_bot"
                    
                    chat_response = await content_service.chat_with_content(
                        user_id=eval_user_id,
                        session_id=eval_session_id,
                        message=question
                    )
                    
                    if isinstance(chat_response, dict):
                        answer = chat_response.get('answer', '')
                        if not answer:
                            answer = chat_response.get('response', '') or chat_response.get('final_answer', '')
                    elif hasattr(chat_response, 'final_answer'):
                        answer = chat_response.final_answer
                    else:
                        answer = str(chat_response)
                    
                    # Update progress - answer generated, now evaluating
                    if question_id and set_id:
                        key = f"{set_id}_{question_id}"
                        if key in progress.questions:
                            progress.questions[key].status = QuestionStatus.EVALUATING
                            logger.info(f"Task {task_id}: Evaluating Q{question_id} in {set_id}")
                    
                    return answer if answer else "No answer generated."
                    
                except Exception as e:
                    logger.error(f"Error generating answer for '{question}': {str(e)}")
                    if question_id and set_id:
                        key = f"{set_id}_{question_id}"
                        if key in progress.questions:
                            progress.questions[key].fail(str(e))
                    return f"Error generating answer: {str(e)}"
            
            async def query_with_metadata(question: str) -> str:
                for q_id, s_id, q_text in questions:
                    if q_text == question:
                        return await query_function(question, q_id, s_id)
                return await query_function(question)
            
            async def on_question_complete(q_id, s_id, status):
                self._update_question_progress(task_id, q_id, s_id, status)
            
            result = await validation_service.validate_from_csv(
                csv_path=str(golden_dataset_path),
                query_function=query_with_metadata,
                max_concurrent=settings.EVALUATION_MAX_CONCURRENT,
                on_question_complete=on_question_complete
            )
            
            if task_id in self.task_progress:
                self.task_progress[task_id].completed_at = datetime.now()
                if isinstance(result, dict) and 'summary' in result:
                    self.task_progress[task_id].results_file = result['summary'].get('results_file')
            
            logger.info(f"Evaluation task {task_id} completed successfully")
            return result
            
        except asyncio.CancelledError:
            logger.warning(f"Evaluation task {task_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Evaluation task {task_id} failed: {str(e)}", exc_info=True)
            raise
    
    def _cleanup_task(self, task_id: str, task: asyncio.Task):
        """Clean up completed tasks."""
        try:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            if task.cancelled():
                logger.info(f"Task {task_id} was cancelled")
            elif task.exception():
                logger.error(f"Task {task_id} failed with exception: {task.exception()}")
            else:
                logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            logger.error(f"Error cleaning up task {task_id}: {str(e)}")
    
    def _update_question_progress(self, task_id: str, question_id: str, set_id: str, status: str):
        """Update progress for a specific question."""
        if task_id in self.task_progress:
            key = f"{set_id}_{question_id}"
            if key in self.task_progress[task_id].questions:
                q_progress = self.task_progress[task_id].questions[key]
                if status == "completed":
                    q_progress.complete()
                    logger.info(f"Task {task_id}: Completed Q{question_id} in {set_id}")
                elif status == "failed":
                    q_progress.fail("Evaluation failed")
                    logger.info(f"Task {task_id}: Failed Q{question_id} in {set_id}")
    
    def get_active_task_count(self) -> int:
        """Get the number of active evaluation tasks."""
        completed_ids = [
            task_id for task_id, task in self.active_tasks.items() 
            if task.done()
        ]
        for task_id in completed_ids:
            del self.active_tasks[task_id]
        
        return len(self.active_tasks)
    
    def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress for a specific task."""
        if task_id in self.task_progress:
            return self.task_progress[task_id].get_status_summary()
        return None
    
    def get_all_progress(self) -> Dict[str, Any]:
        """Get progress for all tasks."""
        return {
            "active_tasks": self.get_active_task_count(),
            "tasks": {
                task_id: progress.get_status_summary() 
                for task_id, progress in self.task_progress.items()
            }
        }
    
    async def run_csv_evaluation(self, csv_path: Path, use_generated_answers: bool = False) -> str:
        """Run CSV evaluation in an isolated task."""
        task_id = str(uuid.uuid4())
        
        task = asyncio.create_task(
            self._run_csv_evaluation_isolated(csv_path, use_generated_answers, task_id)
        )
        
        self.active_tasks[task_id] = task
        task.add_done_callback(lambda t: self._cleanup_task(task_id, t))
        
        return task_id
    
    async def _run_csv_evaluation_isolated(self, csv_path: Path, use_generated_answers: bool, task_id: str):
        """Run CSV evaluation in an isolated context."""
        try:
            logger.info(f"Starting CSV evaluation task {task_id}")
            
            # Import and create fresh service instances
            from .llm_validation_service import LLMValidationService
            from .content_service import ContentService
            from core.config import settings
            
            validation_service = LLMValidationService()
            
            query_function = None
            if not use_generated_answers:
                content_service = ContentService()
                
                async def query_function(question: str) -> str:
                    try:
                        eval_session_id = f"eval_batch_{uuid.uuid4()}"
                        eval_user_id = "evaluation_bot"
                        
                        chat_response = await content_service.chat_with_content(
                            user_id=eval_user_id,
                            session_id=eval_session_id,
                            message=question
                        )
                        
                        if isinstance(chat_response, dict):
                            answer = chat_response.get('answer', '')
                            if not answer:
                                answer = chat_response.get('response', '') or chat_response.get('final_answer', '')
                        else:
                            answer = str(chat_response)
                        
                        return answer if answer else "No answer generated."
                        
                    except Exception as e:
                        logger.error(f"Error generating answer: {str(e)}")
                        return f"Error generating answer: {str(e)}"
            
            result = await validation_service.validate_from_csv(
                csv_path=str(csv_path),
                query_function=query_function,
                max_concurrent=settings.EVALUATION_MAX_CONCURRENT
            )
            
            logger.info(f"CSV evaluation task {task_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"CSV evaluation task {task_id} failed: {str(e)}", exc_info=True)
            raise
    
    async def run_conversation_evaluation(
        self, 
        csv_path: Path, 
        evaluation_mode: str = "contextual",
        max_concurrent_conversations: int = 3
    ) -> str:
        """Run conversation evaluation in an isolated task."""
        task_id = str(uuid.uuid4())
        
        task = asyncio.create_task(
            self._run_conversation_evaluation_isolated(
                csv_path, 
                evaluation_mode, 
                max_concurrent_conversations, 
                task_id
            )
        )
        
        self.active_tasks[task_id] = task
        task.add_done_callback(lambda t: self._cleanup_task(task_id, t))
        
        return task_id
    
    async def _run_conversation_evaluation_isolated(
        self, 
        csv_path: Path, 
        evaluation_mode: str,
        max_concurrent: int,
        task_id: str
    ):
        """Run conversation evaluation in an isolated context."""
        try:
            logger.info(f"Starting conversation evaluation task {task_id}")
            
            # Import and create fresh service instances
            from .llm_validation_service import LLMValidationService
            from utils.conversation_csv_parser import ConversationCSVParser
            from models.conversation_evaluation import EvaluationMode
            
            # Parse conversations first to get count
            conversations = ConversationCSVParser.parse_csv_file(csv_path)
            
            # Initialize progress tracking
            progress = ConversationEvaluationProgress(
                task_id=task_id,
                total_conversations=len(conversations),
                evaluation_mode=evaluation_mode
            )
            
            # Create conversation progress entries
            for conv_id, turns in conversations.items():
                conv_progress = ConversationProgress(
                    conversation_id=conv_id,
                    total_turns=len([t for t in turns if t.role == "user" and t.expected_response])
                )
                # Add turn progress entries
                for turn in turns:
                    if turn.role == "user" and turn.expected_response:
                        conv_progress.turn_progress[turn.turn_number] = ConversationTurnProgress(
                            turn_number=turn.turn_number,
                            message=turn.message[:50] + "..." if len(turn.message) > 50 else turn.message
                        )
                progress.conversations[conv_id] = conv_progress
            
            self.conversation_progress[task_id] = progress
            
            # Create validation service
            validation_service = LLMValidationService()
            
            # Override the conversation callback to update progress
            original_validate = validation_service.validate_conversation
            
            async def validate_with_progress(*args, **kwargs):
                # Hook into the validation process to update progress
                # This is a bit hacky but preserves the isolation pattern
                results = []
                
                # Process each conversation
                for conv_id in conversations:
                    if task_id in self.conversation_progress:
                        conv_prog = self.conversation_progress[task_id].conversations.get(conv_id)
                        if conv_prog:
                            conv_prog.start()
                            logger.info(f"Task {task_id}: Starting conversation {conv_id}")
                
                # Run the actual validation
                results = await original_validate(
                    str(csv_path),
                    EvaluationMode(evaluation_mode),
                    max_concurrent=1  # Process conversations sequentially
                )
                
                # Update progress for completed conversations
                for result in results:
                    if task_id in self.conversation_progress:
                        conv_prog = self.conversation_progress[task_id].conversations.get(result.conversation_id)
                        if conv_prog:
                            if result.overall_rating > 0:
                                conv_prog.complete()
                            else:
                                conv_prog.fail("Evaluation failed")
                            conv_prog.completed_turns = len(result.turn_evaluations)
                
                return results
            
            # Run validation
            results = await validate_with_progress()
            
            # Update task completion
            if task_id in self.conversation_progress:
                self.conversation_progress[task_id].completed_at = datetime.now()
                if results:
                    # Find the most recent conversation validation file
                    import glob
                    result_files = glob.glob("tests/validation_results/conversation_validation_*.json")
                    if result_files:
                        # Get the most recently created file
                        latest_file = max(result_files, key=os.path.getmtime)
                        self.conversation_progress[task_id].results_file = latest_file
                    else:
                        self.conversation_progress[task_id].results_file = None
            
            logger.info(f"Conversation evaluation task {task_id} completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Conversation evaluation task {task_id} failed: {str(e)}", exc_info=True)
            if task_id in self.conversation_progress:
                # Mark all pending conversations as failed
                for conv_prog in self.conversation_progress[task_id].conversations.values():
                    if conv_prog.status == ConversationStatus.PENDING:
                        conv_prog.fail(str(e))
            raise
    
    def get_conversation_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress for a conversation evaluation task."""
        if task_id in self.conversation_progress:
            return self.conversation_progress[task_id].get_status_summary()
        return None
    
    def _update_conversation_turn_progress(
        self, 
        task_id: str, 
        conversation_id: str, 
        turn_number: int, 
        status: str
    ):
        """Update progress for a specific conversation turn."""
        if task_id in self.conversation_progress:
            conv_prog = self.conversation_progress[task_id].conversations.get(conversation_id)
            if conv_prog and turn_number in conv_prog.turn_progress:
                turn_prog = conv_prog.turn_progress[turn_number]
                if status == "generating":
                    turn_prog.status = QuestionStatus.GENERATING_ANSWER
                    conv_prog.current_turn = turn_number
                elif status == "evaluating":
                    turn_prog.status = QuestionStatus.EVALUATING
                elif status == "completed":
                    turn_prog.status = QuestionStatus.COMPLETED
                    conv_prog.completed_turns += 1
                elif status == "failed":
                    turn_prog.status = QuestionStatus.FAILED


# Global instance for the application
evaluation_runner = EvaluationRunner()