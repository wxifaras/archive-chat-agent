from dataclasses import dataclass, field
from typing import List, Set, Optional
from typing import Literal
from services.azure_ai_search_service import SearchResult
from pydantic import BaseModel

class SearchPromptResponse(BaseModel):
    search_query: str
    filter: str

@dataclass
class ContentConversation:
    user_query: str
    user_id: str 
    session_id: str
    use_agentic_retrieval: bool
    max_attempts: int = 3
    
    # State maintained while processing the agent rag workflow
    attempts: int = field(default=0)
    # this should probably have it's own dictionary type; we can update this later
    search_history: List[dict] = field(default_factory=list)
    current_results: List[SearchResult] = field(default_factory=list)
    vetted_results: List[SearchResult] = field(default_factory=list)
    discarded_results: List[SearchResult] = field(default_factory=list)
    processed_ids: Set[str] = field(default_factory=set)
    thought_process: List[dict] = field(default_factory=list)
    reviews: List[str] = field(default_factory=list)      # Thought processes from reviews
    decisions: List[str] = field(default_factory=list)    # Store the actual decisions
    sub_queries: List[List[dict]] = field(default_factory=list)  # Store sub-queries from agentic retrieval
    
    def should_continue(self) -> bool:
        return self.attempts < self.max_attempts and not self.has_sufficient_results()
    
    def has_sufficient_results(self) -> bool:
        return "finalize" in self.decisions
    
    def has_search_history(self) -> bool:
        return len(self.search_history) > 0
    
    def has_valid_results(self) -> bool:
        return len(self.vetted_results) > 0
    
    def add_search_attempt(self, query: str):
        self.attempts += 1
        self.search_history.append({
            "query": query
        })
    
    def to_result(self, final_answer: str) -> 'ConversationResult':
        """Convert conversation to final result"""
        return ConversationResult(
            final_answer=final_answer,
            citations=self.vetted_results,
            thought_process=self.thought_process,
            attempts=self.attempts,
            search_queries=[search["query"] for search in self.search_history],
            sub_queries=self.sub_queries
        )
    
@dataclass
class ConversationResult:
    final_answer: str
    citations: List[SearchResult]
    thought_process: List[dict]
    attempts: int
    search_queries: List[str]
    sub_queries: List[List[dict]]

NUM_SEARCH_RESULTS = 5

# Create a type for indices from 0 to NUM_SEARCH_RESULTS-1
#SearchResultIndex = Literal[0, 1, 2, 3, 4]
# Create a type for indices from 0 to 49 (50 possible results)
SearchResultIndex = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49
]

class ReviewDecision(BaseModel):
    """Schema for review agent decisions"""
    thought_process: str
    valid_results: List[SearchResultIndex]  # Indices of valid results
    invalid_results: List[SearchResultIndex]  # Indices of invalid results
    decision: Literal["retry", "finalize"]