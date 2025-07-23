"""
Pydantic models for LLM evaluation and validation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class EvaluationRequest(BaseModel):
    """Request model for single Q&A evaluation"""
    question: str = Field(..., description="The question asked")
    ground_truth: str = Field(..., description="The expected/ground truth answer")
    generated_answer: str = Field(..., description="The LLM-generated answer to evaluate")


class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
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


class BatchEvaluationRequest(BaseModel):
    """Request model for batch evaluation from CSV"""
    csv_file_path: str = Field(..., description="Path to CSV file containing Q&A pairs")
    use_generated_answers: bool = Field(
        default=False, 
        description="If true, expects 'generated_answer' column in CSV. If false, generates answers using the chat endpoint."
    )




class EvaluationSummary(BaseModel):
    """Summary statistics for batch evaluation"""
    total_evaluations: int = Field(..., description="Total number of evaluations performed")
    five_star_count: int = Field(..., description="Number of 5-star ratings")
    three_star_count: int = Field(..., description="Number of 3-star ratings")
    one_star_count: int = Field(..., description="Number of 1-star ratings")
    five_star_percentage: float = Field(..., description="Percentage of 5-star ratings")
    three_star_percentage: float = Field(..., description="Percentage of 3-star ratings")
    one_star_percentage: float = Field(..., description="Percentage of 1-star ratings")
    results_file: Optional[str] = Field(None, description="Path to detailed results JSON file")


class BatchEvaluationResponse(BaseModel):
    """Response model for batch evaluation results"""
    results: List[EvaluationResponse] = Field(..., description="List of individual evaluation results")
    summary: EvaluationSummary = Field(..., description="Summary statistics of the evaluation")


class EvaluationStatus(BaseModel):
    """Status response for async evaluation tasks"""
    status: str = Field(..., description="Status of the evaluation task")
    message: str = Field(..., description="Status message")
    task_id: Optional[str] = Field(None, description="Task ID for tracking async operations")
    file_path: Optional[str] = Field(None, description="Path to the file being processed")