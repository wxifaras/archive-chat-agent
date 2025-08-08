"""
Models for tracking evaluation progress with detailed question-level and conversation-level information.
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


class ConversationStatus(str, Enum):
    """Status of conversation evaluation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
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


class ConversationTurnProgress(BaseModel):
    """Progress tracking for a single turn in a conversation"""
    turn_number: int = Field(..., description="Turn number in conversation")
    status: QuestionStatus = Field(default=QuestionStatus.PENDING)
    message: str = Field(..., description="Turn message preview")
    

class ConversationProgress(BaseModel):
    """Progress tracking for a single conversation"""
    conversation_id: str = Field(..., description="Unique conversation ID")
    total_turns: int = Field(..., description="Total number of turns")
    completed_turns: int = Field(default=0)
    current_turn: Optional[int] = None
    status: ConversationStatus = Field(default=ConversationStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    turn_progress: Dict[int, ConversationTurnProgress] = Field(default_factory=dict)
    
    def start(self):
        """Mark conversation as started"""
        self.status = ConversationStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete(self):
        """Mark conversation as completed"""
        self.status = ConversationStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def fail(self, error: str):
        """Mark conversation as failed"""
        self.status = ConversationStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
    
    @property
    def progress_percentage(self) -> float:
        """Calculate conversation completion percentage"""
        if self.total_turns == 0:
            return 0.0
        return (self.completed_turns / self.total_turns) * 100


class ConversationEvaluationProgress(BaseModel):
    """Progress tracking for conversation evaluations"""
    task_id: str = Field(..., description="Unique task ID")
    total_conversations: int = Field(..., description="Total number of conversations")
    conversations: Dict[str, ConversationProgress] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results_file: Optional[str] = None
    evaluation_mode: str = Field(default="contextual")
    
    @property
    def completed_count(self) -> int:
        """Number of completed conversations"""
        return sum(1 for c in self.conversations.values() if c.status == ConversationStatus.COMPLETED)
    
    @property
    def failed_count(self) -> int:
        """Number of failed conversations"""
        return sum(1 for c in self.conversations.values() if c.status == ConversationStatus.FAILED)
    
    @property
    def in_progress_count(self) -> int:
        """Number of conversations in progress"""
        return sum(1 for c in self.conversations.values() 
                  if c.status in [ConversationStatus.IN_PROGRESS, ConversationStatus.EVALUATING])
    
    @property
    def progress_percentage(self) -> float:
        """Overall progress percentage"""
        if self.total_conversations == 0:
            return 0.0
        return ((self.completed_count + self.failed_count) / self.total_conversations) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if all conversations are evaluated"""
        return (self.completed_count + self.failed_count) == self.total_conversations
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get detailed status summary"""
        return {
            "task_id": self.task_id,
            "evaluation_type": "conversation",
            "evaluation_mode": self.evaluation_mode,
            "total_conversations": self.total_conversations,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "in_progress": self.in_progress_count,
            "progress_percentage": round(self.progress_percentage, 1),
            "is_complete": self.is_complete,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results_file": self.results_file,
            "current_conversations": [
                {
                    "conversation_id": c.conversation_id,
                    "status": c.status,
                    "progress": f"{c.completed_turns}/{c.total_turns} turns",
                    "current_turn": c.current_turn
                }
                for c in self.conversations.values()
                if c.status in [ConversationStatus.IN_PROGRESS, ConversationStatus.EVALUATING]
            ][:5]  # Show up to 5 current conversations
        }