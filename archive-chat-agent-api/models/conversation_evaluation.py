"""
Models for multi-turn conversation evaluation.
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import StrEnum


class ConversationRole(StrEnum):
    """Roles in a conversation"""
    USER = "user"
    ASSISTANT = "assistant"


class EvaluationMode(StrEnum):
    """Evaluation modes for conversations"""
    CONTEXTUAL = "contextual"  # Evaluate with full context consideration
    TURN_BY_TURN = "turn_by_turn"  # Evaluate each turn independently
    HOLISTIC = "holistic"  # Evaluate entire conversation as a whole


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation for evaluation"""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    turn_number: int = Field(..., description="Sequential turn number in the conversation", ge=1)
    role: ConversationRole = Field(..., description="Role of the speaker (user or assistant)")
    message: str = Field(..., description="The message content")
    expected_response: Optional[str] = Field(
        None, 
        description="Expected response for user turns (ground truth)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the turn"
    )
    
    @field_validator('expected_response')
    def validate_expected_response(cls, v, values):
        """Ensure expected_response is only set for user turns"""
        if 'role' in values.data and values.data['role'] == ConversationRole.ASSISTANT and v is not None:
            raise ValueError("expected_response should only be set for user turns")
        return v
    
    class Config:
        use_enum_values = True


class ConversationEvaluationRequest(BaseModel):
    """Request to evaluate a multi-turn conversation"""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    turns: List[ConversationTurn] = Field(
        ..., 
        description="List of conversation turns to evaluate",
        min_items=1
    )
    evaluation_mode: EvaluationMode = Field(
        default=EvaluationMode.CONTEXTUAL,
        description="How to evaluate the conversation"
    )
    max_turns_to_evaluate: Optional[int] = Field(
        None,
        description="Maximum number of turns to evaluate (useful for long conversations)",
        ge=1
    )
    
    @field_validator('turns')
    def validate_turn_order(cls, v):
        """Ensure turns are in sequential order"""
        turn_numbers = [turn.turn_number for turn in v]
        expected_numbers = list(range(1, len(v) + 1))
        
        # Check if turn numbers are sequential
        if sorted(turn_numbers) != expected_numbers:
            raise ValueError("Turn numbers must be sequential starting from 1")
        
        # Ensure turns are ordered by turn_number
        for i, turn in enumerate(v):
            if turn.turn_number != i + 1:
                raise ValueError(f"Turn at index {i} should have turn_number {i + 1}, got {turn.turn_number}")
        
        return v
    
    @field_validator('turns')
    def validate_conversation_consistency(cls, v):
        """Ensure all turns belong to the same conversation"""
        if not v:
            return v
            
        conversation_ids = set(turn.conversation_id for turn in v)
        if len(conversation_ids) > 1:
            raise ValueError("All turns must belong to the same conversation_id")
        
        return v


class TurnEvaluation(BaseModel):
    """Evaluation result for a single conversation turn"""
    turn_number: int = Field(..., description="Turn number being evaluated")
    question: str = Field(..., description="The user's question")
    expected_response: str = Field(..., description="Expected response (ground truth)")
    generated_response: str = Field(..., description="System's generated response")
    rating: Literal[1, 3, 5] = Field(..., description="Rating for this turn")
    correctness_assessment: str = Field(..., description="Assessment of answer correctness")
    context_usage_assessment: str = Field(..., description="Assessment of context usage")
    evaluation_thoughts: str = Field(..., description="Detailed evaluation thoughts")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMetrics(BaseModel):
    """Detailed metrics for conversation quality"""
    context_coherence_score: float = Field(
        ..., 
        description="How well context is maintained (0-1)",
        ge=0.0,
        le=1.0
    )
    follow_up_accuracy: float = Field(
        ...,
        description="Accuracy in handling follow-up questions (0-1)",
        ge=0.0,
        le=1.0
    )
    information_consistency: float = Field(
        ...,
        description="Consistency of information across turns (0-1)", 
        ge=0.0,
        le=1.0
    )
    conversation_flow_score: float = Field(
        ...,
        description="Natural flow of conversation (0-1)",
        ge=0.0,
        le=1.0
    )
    overall_effectiveness: float = Field(
        ...,
        description="Overall conversation effectiveness (0-1)",
        ge=0.0,
        le=1.0
    )
    
    @property
    def average_score(self) -> float:
        """Calculate average of all metrics"""
        scores = [
            self.context_coherence_score,
            self.follow_up_accuracy,
            self.information_consistency,
            self.conversation_flow_score,
            self.overall_effectiveness
        ]
        return sum(scores) / len(scores)


class ConversationEvaluationResult(BaseModel):
    """Complete evaluation result for a multi-turn conversation"""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    session_id: str = Field(..., description="Session ID used for this evaluation")
    evaluation_mode: EvaluationMode = Field(..., description="Evaluation mode used")
    overall_rating: Literal[1, 3, 5] = Field(..., description="Overall conversation rating")
    turn_evaluations: List[TurnEvaluation] = Field(
        ...,
        description="Individual turn evaluation results"
    )
    conversation_metrics: ConversationMetrics = Field(
        ...,
        description="Detailed conversation quality metrics"
    )
    overall_evaluation_thoughts: str = Field(
        ...,
        description="High-level evaluation thoughts for entire conversation"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Identified strengths in the conversation"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Identified weaknesses in the conversation"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the evaluation was performed"
    )
    evaluation_duration_seconds: Optional[float] = Field(
        None,
        description="Time taken to complete evaluation"
    )
    
    @property
    def average_turn_rating(self) -> float:
        """Calculate average rating across all turns"""
        if not self.turn_evaluations:
            return 0.0
        return sum(turn.rating for turn in self.turn_evaluations) / len(self.turn_evaluations)
    
    @property
    def turn_success_rate(self) -> float:
        """Percentage of turns rated 5 stars"""
        if not self.turn_evaluations:
            return 0.0
        successful_turns = sum(1 for turn in self.turn_evaluations if turn.rating == 5)
        return (successful_turns / len(self.turn_evaluations)) * 100
    
    class Config:
        use_enum_values = True


class ConversationBatchEvaluationRequest(BaseModel):
    """Request to evaluate multiple conversations from a CSV file"""
    csv_file_path: str = Field(..., description="Path to CSV file containing conversations")
    evaluation_mode: EvaluationMode = Field(
        default=EvaluationMode.CONTEXTUAL,
        description="How to evaluate conversations"
    )
    max_concurrent_conversations: int = Field(
        default=3,
        description="Maximum concurrent conversation evaluations",
        ge=1,
        le=10
    )
    continue_on_error: bool = Field(
        default=True,
        description="Whether to continue if a conversation evaluation fails"
    )


class ConversationBatchEvaluationResult(BaseModel):
    """Results from evaluating multiple conversations"""
    total_conversations: int = Field(..., description="Total number of conversations processed")
    successful_evaluations: int = Field(..., description="Number of successful evaluations")
    failed_evaluations: int = Field(..., description="Number of failed evaluations") 
    results: List[ConversationEvaluationResult] = Field(
        default_factory=list,
        description="Individual conversation results"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Errors by conversation_id"
    )
    overall_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregate metrics across all conversations"
    )
    evaluation_summary_file: Optional[str] = Field(
        None,
        description="Path to detailed summary file"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate evaluation success rate"""
        if self.total_conversations == 0:
            return 0.0
        return (self.successful_evaluations / self.total_conversations) * 100


class ConversationCSVRow(BaseModel):
    """Schema for conversation CSV row"""
    conversation_id: str
    turn_number: int
    role: str
    message: str
    expected_response: Optional[str] = None
    
    @field_validator('role')
    def validate_role(cls, v):
        """Ensure role is valid"""
        if v.lower() not in ['user', 'assistant']:
            raise ValueError(f"Invalid role: {v}. Must be 'user' or 'assistant'")
        return v.lower()
    
    def to_conversation_turn(self) -> ConversationTurn:
        """Convert CSV row to ConversationTurn"""
        return ConversationTurn(
            conversation_id=self.conversation_id,
            turn_number=self.turn_number,
            role=ConversationRole(self.role),
            message=self.message,
            expected_response=self.expected_response
        )