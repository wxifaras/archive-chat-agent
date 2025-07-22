"""
API routes for LLM response evaluation and validation.
"""

import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from pathlib import Path
import tempfile
import shutil
import httpx
import uuid

from models.evaluation import (
    EvaluationRequest,
    EvaluationResponse,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationStatus
)
from services.llm_validation_service import LLMValidationService
from services.evaluation_runner import evaluation_runner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])

@router.post("/single", response_model=EvaluationResponse)
async def evaluate_single_qa(request: EvaluationRequest):
    """
    Evaluate a single question-answer pair against ground truth.
    
    This endpoint performs synchronous evaluation of a single Q&A pair.
    """
    try:
        validation_service = LLMValidationService()
        
        result = await validation_service.validate_single_qa(
            question=request.question,
            ground_truth=request.ground_truth,
            generated_answer=request.generated_answer
        )
        
        return EvaluationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in single Q&A evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=EvaluationStatus)
async def evaluate_batch_csv(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Evaluate multiple Q&A pairs from a CSV file.
    
    CSV format expected:
    - question: The question to ask
    - expected_answer: The ground truth answer
    - generated_answer: (optional) Pre-generated answer to evaluate
    
    If generated_answer column is not present, the system will generate answers
    using the chat endpoint.
    """
    try:
        # Verify file exists
        csv_path = Path(request.csv_file_path)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV file not found: {request.csv_file_path}")
        
        # Use the evaluation runner to handle the task properly
        task_id = await evaluation_runner.run_csv_evaluation(
            csv_path=csv_path,
            use_generated_answers=request.use_generated_answers
        )
        
        return EvaluationStatus(
            status="started",
            message="Batch evaluation started in background",
            task_id=task_id,
            file_path=str(csv_path)
        )
        
    except Exception as e:
        logger.error(f"Error starting batch evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-csv", response_model=EvaluationStatus)
async def evaluate_uploaded_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    use_generated_answers: bool = False
):
    """
    Upload and evaluate a CSV file containing Q&A pairs.
    
    CSV format expected:
    - question: The question to ask
    - expected_answer: The ground truth answer
    - generated_answer: (optional) Pre-generated answer to evaluate
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Create request and process
        request = BatchEvaluationRequest(
            csv_file_path=tmp_path,
            use_generated_answers=use_generated_answers
        )
        
        return await evaluate_batch_csv(request, background_tasks)
        
    except Exception as e:
        logger.error(f"Error processing uploaded CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))






@router.get("/status")
async def get_evaluation_status():
    """
    Get the status of all evaluation tasks with detailed progress.
    """
    progress = evaluation_runner.get_all_progress()
    
    # For backward compatibility, include active_evaluations count
    progress["active_evaluations"] = progress["active_tasks"]
    progress["message"] = f"{progress['active_tasks']} evaluation(s) currently running"
    
    return progress


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get detailed status for a specific evaluation task.
    """
    progress = evaluation_runner.get_task_progress(task_id)
    
    if not progress:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return progress


@router.post("/golden-dataset", response_model=EvaluationStatus)
async def evaluate_golden_dataset(background_tasks: BackgroundTasks):
    """
    Evaluate the system against the pre-configured golden dataset.
    
    This endpoint uses the golden dataset located at:
    data/GoldenDataSet_Template_CSV/golden_dataset_template.csv
    
    It will generate answers using the chat service and evaluate them
    against the ground truth answers in the golden dataset.
    """
    try:
        # Path to the golden dataset
        golden_dataset_path = Path("tests/data/golden_dataset_template.csv")
        
        if not golden_dataset_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Golden dataset not found at: {golden_dataset_path}"
            )
        
        # Use the evaluation runner to handle the task properly
        task_id = await evaluation_runner.run_golden_dataset_evaluation(golden_dataset_path)
        
        return EvaluationStatus(
            status="started",
            message="Golden dataset evaluation started in background",
            task_id=task_id,
            file_path=str(golden_dataset_path)
        )
        
    except Exception as e:
        logger.error(f"Error starting golden dataset evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))