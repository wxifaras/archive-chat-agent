"""
Models for tracking evaluation progress with detailed question-level information.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class QuestionStatus(str, Enum):
    """Status of individual question evaluation"""
    PENDING = "pending"
    GENERATING_ANSWER = "generating_answer"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionProgress(BaseModel):
    """Progress tracking for a single question"""
    question_id: str = Field(..., description="Unique ID of the question")
    set_id: str = Field(..., description="ID of the question set")
    question: str = Field(..., description="The actual question text")
    status: QuestionStatus = Field(default=QuestionStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def start(self):
        """Mark question as started"""
        self.status = QuestionStatus.GENERATING_ANSWER
        self.started_at = datetime.now()
    
    def complete(self):
        """Mark question as completed"""
        self.status = QuestionStatus.COMPLETED
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def fail(self, error: str):
        """Mark question as failed"""
        self.status = QuestionStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class EvaluationProgress(BaseModel):
    """Overall evaluation progress tracking"""
    task_id: str = Field(..., description="Unique task ID")
    total_questions: int = Field(..., description="Total number of questions")
    questions: Dict[str, QuestionProgress] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results_file: Optional[str] = None
    
    @property
    def completed_count(self) -> int:
        """Number of completed questions"""
        return sum(1 for q in self.questions.values() if q.status == QuestionStatus.COMPLETED)
    
    @property
    def failed_count(self) -> int:
        """Number of failed questions"""
        return sum(1 for q in self.questions.values() if q.status == QuestionStatus.FAILED)
    
    @property
    def in_progress_count(self) -> int:
        """Number of questions currently being processed"""
        return sum(1 for q in self.questions.values() if q.status in [QuestionStatus.GENERATING_ANSWER, QuestionStatus.EVALUATING])
    
    @property
    def pending_count(self) -> int:
        """Number of pending questions"""
        return sum(1 for q in self.questions.values() if q.status == QuestionStatus.PENDING)
    
    @property
    def progress_percentage(self) -> float:
        """Completion percentage"""
        if self.total_questions == 0:
            return 0.0
        return ((self.completed_count + self.failed_count) / self.total_questions) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if evaluation is complete"""
        return (self.completed_count + self.failed_count) == self.total_questions
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get detailed status summary"""
        return {
            "task_id": self.task_id,
            "total_questions": self.total_questions,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "in_progress": self.in_progress_count,
            "pending": self.pending_count,
            "progress_percentage": round(self.progress_percentage, 1),
            "is_complete": self.is_complete,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results_file": self.results_file,
            "current_questions": [
                {
                    "question_id": q.question_id,
                    "set_id": q.set_id,
                    "status": q.status,
                    "question": q.question[:100] + "..." if len(q.question) > 100 else q.question
                }
                for q in self.questions.values() 
                if q.status in [QuestionStatus.GENERATING_ANSWER, QuestionStatus.EVALUATING]
            ][:5]  # Show up to 5 current questions
        }