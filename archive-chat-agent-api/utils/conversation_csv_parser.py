"""
CSV parser for multi-turn conversation evaluation data.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Union, TextIO, Optional, Tuple
from collections import defaultdict

from models.conversation_evaluation import (
    ConversationCSVRow,
    ConversationTurn,
    ConversationEvaluationRequest,
    ConversationRole
)

logger = logging.getLogger(__name__)


class ConversationCSVParser:
    """Parser for conversation CSV files."""
    
    REQUIRED_COLUMNS = {
        'conversation_id',
        'turn_number', 
        'role',
        'message'
    }
    
    OPTIONAL_COLUMNS = {
        'expected_response'
    }
    
    @classmethod
    def parse_csv_file(
        cls,
        file_path: Union[str, Path]
    ) -> Dict[str, List[ConversationTurn]]:
        """
        Parse a CSV file containing conversation data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary mapping conversation_id to list of ConversationTurn objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Parsing conversation CSV file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.parse_csv_stream(f)
    
    @classmethod
    def parse_csv_stream(
        cls,
        csv_stream: TextIO
    ) -> Dict[str, List[ConversationTurn]]:
        """
        Parse a CSV stream containing conversation data.
        
        Args:
            csv_stream: File-like object containing CSV data
            
        Returns:
            Dictionary mapping conversation_id to list of ConversationTurn objects
            
        Raises:
            ValueError: If CSV format is invalid
        """
        reader = csv.DictReader(csv_stream)
        
        # Validate columns
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no headers")
        
        fieldnames_set = set(reader.fieldnames)
        missing_columns = cls.REQUIRED_COLUMNS - fieldnames_set
        
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}. "
                f"Found columns: {', '.join(reader.fieldnames)}"
            )
        
        # Parse rows
        conversations: Dict[str, List[ConversationCSVRow]] = defaultdict(list)
        errors = []
        row_count = 0
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            row_count += 1
            
            try:
                # Clean up the row data
                cleaned_row = {k: v.strip() if v else '' for k, v in row.items()}
                
                # Validate and parse the row
                csv_row = ConversationCSVRow(
                    conversation_id=cleaned_row.get('conversation_id', ''),
                    turn_number=int(cleaned_row.get('turn_number', 0)),
                    role=cleaned_row.get('role', ''),
                    message=cleaned_row.get('message', ''),
                    expected_response=cleaned_row.get('expected_response') or None
                )
                
                # Additional validation
                if not csv_row.conversation_id:
                    errors.append(f"Row {row_num}: Empty conversation_id")
                    continue
                    
                if not csv_row.message:
                    errors.append(f"Row {row_num}: Empty message")
                    continue
                
                conversations[csv_row.conversation_id].append(csv_row)
                
            except ValueError as e:
                errors.append(f"Row {row_num}: {str(e)}")
            except Exception as e:
                errors.append(f"Row {row_num}: Unexpected error: {str(e)}")
        
        if errors:
            error_summary = "\n".join(errors[:10])  # Show first 10 errors
            if len(errors) > 10:
                error_summary += f"\n... and {len(errors) - 10} more errors"
            raise ValueError(f"CSV parsing errors:\n{error_summary}")
        
        if row_count == 0:
            raise ValueError("CSV file contains no data rows")
        
        logger.info(f"Parsed {row_count} rows into {len(conversations)} conversations")
        
        # Convert to ConversationTurn objects and sort by turn number
        result = {}
        
        for conv_id, csv_rows in conversations.items():
            # Sort by turn number
            csv_rows.sort(key=lambda x: x.turn_number)
            
            # Validate turn sequence
            expected_turns = list(range(1, len(csv_rows) + 1))
            actual_turns = [row.turn_number for row in csv_rows]
            
            if actual_turns != expected_turns:
                raise ValueError(
                    f"Conversation {conv_id} has invalid turn sequence. "
                    f"Expected: {expected_turns}, Got: {actual_turns}"
                )
            
            # Convert to ConversationTurn objects
            turns = [row.to_conversation_turn() for row in csv_rows]
            
            # Validate conversation structure
            cls._validate_conversation_structure(conv_id, turns)
            
            result[conv_id] = turns
        
        return result
    
    @classmethod
    def _validate_conversation_structure(
        cls,
        conversation_id: str,
        turns: List[ConversationTurn]
    ) -> None:
        """
        Validate the structure of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            turns: List of conversation turns
            
        Raises:
            ValueError: If conversation structure is invalid
        """
        # Check that user turns have expected responses
        for turn in turns:
            if turn.role == ConversationRole.USER and not turn.expected_response:
                logger.warning(
                    f"Conversation {conversation_id}, turn {turn.turn_number}: "
                    f"User turn missing expected_response"
                )
            
            if turn.role == ConversationRole.ASSISTANT and turn.expected_response:
                raise ValueError(
                    f"Conversation {conversation_id}, turn {turn.turn_number}: "
                    f"Assistant turn should not have expected_response"
                )
        
        # Warn about unusual patterns
        user_turns = [t for t in turns if t.role == ConversationRole.USER]
        if len(user_turns) == 0:
            raise ValueError(
                f"Conversation {conversation_id} has no user turns"
            )
    
    @classmethod
    def create_evaluation_requests(
        cls,
        conversations: Dict[str, List[ConversationTurn]],
        evaluation_mode: str = "contextual",
        max_turns_per_conversation: Optional[int] = None
    ) -> List[ConversationEvaluationRequest]:
        """
        Create evaluation request objects from parsed conversations.
        
        Args:
            conversations: Dictionary of conversation_id to turns
            evaluation_mode: How to evaluate conversations
            max_turns_per_conversation: Limit turns per conversation
            
        Returns:
            List of ConversationEvaluationRequest objects
        """
        requests = []
        
        for conv_id, turns in conversations.items():
            # Apply turn limit if specified
            if max_turns_per_conversation and len(turns) > max_turns_per_conversation:
                logger.info(
                    f"Conversation {conv_id}: Limiting from {len(turns)} to "
                    f"{max_turns_per_conversation} turns"
                )
                turns = turns[:max_turns_per_conversation]
            
            request = ConversationEvaluationRequest(
                conversation_id=conv_id,
                turns=turns,
                evaluation_mode=evaluation_mode,
                max_turns_to_evaluate=max_turns_per_conversation
            )
            
            requests.append(request)
        
        return requests
    
    @classmethod
    def write_sample_csv(
        cls,
        file_path: Union[str, Path],
        sample_conversations: Optional[List[Tuple[str, List[Dict]]]] = None
    ) -> None:
        """
        Write a sample CSV file with example conversations.
        
        Args:
            file_path: Where to write the sample
            sample_conversations: Optional custom conversations
        """
        file_path = Path(file_path)
        
        if sample_conversations is None:
            # Default sample conversations
            sample_conversations = [
                ("SAMPLE001", [
                    {
                        "turn_number": 1,
                        "role": "user",
                        "message": "What database does the system use?",
                        "expected_response": "The system uses PostgreSQL as its primary database"
                    },
                    {
                        "turn_number": 2,
                        "role": "user", 
                        "message": "What version?",
                        "expected_response": "PostgreSQL version 15"
                    },
                    {
                        "turn_number": 3,
                        "role": "user",
                        "message": "Is it configured for high availability?",
                        "expected_response": "Yes, it's configured with streaming replication and automatic failover"
                    }
                ]),
                ("SAMPLE002", [
                    {
                        "turn_number": 1,
                        "role": "user",
                        "message": "Explain the authentication system",
                        "expected_response": "The system uses OAuth 2.0 with JWT tokens for authentication"
                    },
                    {
                        "turn_number": 2,
                        "role": "user",
                        "message": "How long are tokens valid?", 
                        "expected_response": "Access tokens are valid for 1 hour, refresh tokens for 30 days"
                    }
                ])
            ]
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['conversation_id', 'turn_number', 'role', 'message', 'expected_response']
            )
            
            writer.writeheader()
            
            for conv_id, turns in sample_conversations:
                for turn in turns:
                    writer.writerow({
                        'conversation_id': conv_id,
                        'turn_number': turn['turn_number'],
                        'role': turn['role'],
                        'message': turn['message'],
                        'expected_response': turn.get('expected_response', '')
                    })
        
        logger.info(f"Sample CSV written to: {file_path}")


def parse_conversation_csv(
    file_path: Union[str, Path]
) -> Dict[str, List[ConversationTurn]]:
    """
    Convenience function to parse a conversation CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary mapping conversation_id to list of turns
    """
    return ConversationCSVParser.parse_csv_file(file_path)


def create_sample_conversation_csv(
    file_path: Union[str, Path]
) -> None:
    """
    Convenience function to create a sample conversation CSV.
    
    Args:
        file_path: Where to write the sample
    """
    ConversationCSVParser.write_sample_csv(file_path)