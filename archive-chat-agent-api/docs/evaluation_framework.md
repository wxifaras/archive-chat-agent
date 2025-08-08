# Archive Chat Agent Evaluation Framework

## Overview

The Archive Chat Agent includes a comprehensive evaluation framework designed to assess the quality and accuracy of the RAG (Retrieval Augmented Generation) pipeline. This framework enables systematic testing of the system's ability to answer questions based on indexed content, comparing generated answers against known ground truth.

## Architecture

### Core Components

1. **LLM Validation Service** (`services/llm_validation_service.py`)
   - Manages evaluation logic using Azure OpenAI
   - Configurable to use a separate GPT model for evaluation (default: gpt-4.1)
   - Implements parallel processing for efficient batch evaluation
   - Generates structured evaluation responses with ratings and reasoning
   - Automatically handles model-specific temperature requirements (e.g., o3 models require temperature=1.0)
   - **NEW**: Supports multi-turn conversation evaluation with context awareness

2. **Evaluation Runner** (`services/evaluation_runner.py`)
   - Orchestrates background evaluation tasks
   - Manages task isolation and progress tracking
   - Integrates with the RAG pipeline to generate answers
   - Handles error recovery and task completion
   - **NEW**: Supports conversation-level progress tracking with turn-by-turn visibility

3. **API Endpoints** (`routes/evaluation.py`)
   - RESTful endpoints for triggering evaluations
   - Support for CSV input format
   - Real-time progress monitoring capabilities
   - Task-based asynchronous processing
   - **NEW**: Dedicated endpoints for multi-turn conversation evaluation

4. **Progress Tracking** (`models/evaluation.py`, `models/evaluation_progress.py`)
   - Detailed question-level status tracking
   - Set-based organization for grouped questions
   - Real-time progress updates during evaluation
   - Comprehensive completion statistics
   - **NEW**: Conversation-level progress with turn-by-turn status tracking

## API Endpoints

### 1. Single Evaluation
**Endpoint:** `POST /api/evaluate/single`

Evaluates a single question-answer pair.

**Request Body:**
```json
{
  "question": "What was the change in the Bain Consumer Health Index?",
  "ground_truth": "The index increased by 0.1 points to 97.7",
  "generated_answer": "The Bain CHI rose by 0.1 to reach 97.7"
}
```

**Response:**
```json
{
  "question": "What was the change in the Bain Consumer Health Index?",
  "ground_truth": "The index increased by 0.1 points to 97.7",
  "answer": "The Bain CHI rose by 0.1 to reach 97.7",
  "thoughts": "The generated answer correctly identifies both the change amount (0.1 points) and final value (97.7)...",
  "stars": 5
}
```

### 2. Batch Evaluation from CSV
**Endpoint:** `POST /api/evaluate/batch`

Evaluates multiple Q&A pairs from a CSV file.

**Request Body:**
```json
{
  "csv_path": "path/to/evaluation_data.csv"
}
```

**Response:**
```json
{
  "status": "Evaluation started in background",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Processing 12 questions from CSV"
}
```

### 3. Golden Dataset Evaluation
**Endpoint:** `POST /api/evaluate/golden-dataset`

Runs evaluation using the pre-configured golden dataset.

**Response:**
```json
{
  "status": "Golden dataset evaluation started",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_questions": 12
}
```

### 4. Task Status Monitoring
**Endpoint:** `GET /api/evaluate/status/{task_id}`

Retrieves the status of a specific evaluation task.

**Response:**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "in_progress",
  "progress": {
    "completed": 8,
    "failed": 1,
    "total": 12,
    "current_set": "Set_5",
    "in_progress_questions": [
      {
        "question_id": "Q1",
        "set_id": "Set_5",
        "question": "According to the social media post...",
        "status": "in_progress"
      }
    ]
  },
  "results_file": null
}
```

### 5. Upload Endpoints
- **CSV Upload:** `POST /api/evaluate/upload-csv`

These endpoints accept file uploads for evaluation.

### 6. Conversation Evaluation (NEW)
**Endpoint:** `POST /api/evaluate/conversation/evaluate`

Evaluates multi-turn conversations with context awareness.

**Request Body:**
```json
{
  "csv_file_path": "tests/data/golden_dataset_conversations.csv",
  "evaluation_mode": "contextual",
  "max_concurrent_conversations": 3
}
```

**CSV Format for Conversations:**
```csv
conversation_id,turn_number,role,message,expected_response
Set_1,1,user,"What database does the system use?","PostgreSQL"
Set_1,2,user,"What version?","Version 15"
```

**Response:**
```json
{
  "status": "started",
  "message": "Conversation evaluation started in background",
  "task_id": "b9393c46-9af7-41a6-9bc4-0aaddf0377fb",
  "file_path": "tests/data/golden_dataset_conversations.csv"
}
```

### 7. Conversation Upload
**Endpoint:** `POST /api/evaluate/conversation/upload`

Upload and evaluate conversation CSV files.

### 8. Conversation Status
**Endpoint:** `GET /api/evaluate/conversation/status/{task_id}`

Get detailed conversation evaluation progress.

**Response:**
```json
{
  "task_id": "b9393c46-9af7-41a6-9bc4-0aaddf0377fb",
  "evaluation_type": "conversation",
  "evaluation_mode": "contextual",
  "total_conversations": 6,
  "completed": 3,
  "in_progress": 1,
  "progress_percentage": 50.0,
  "current_conversations": [
    {
      "conversation_id": "Set_4",
      "status": "in_progress",
      "progress": "1/2 turns",
      "current_turn": 2
    }
  ]
}
```

## Evaluation Scripts

### run_evaluation.py

#### Purpose
This unified script provides a command-line interface to run both golden dataset and conversation evaluations via the API endpoints.

#### Usage
```bash
# Run conversation evaluation (default)
python run_evaluation.py

# Run golden dataset evaluation
python run_evaluation.py --type golden

# With custom conversation CSV
python run_evaluation.py --csv custom.csv --timeout 60
```

#### Features
- Unified interface for all evaluation types
- Command-line arguments for flexibility
- Progress monitoring with detailed output
- Clickable file paths in terminal output
- Automatic environment variable loading from .env


### How It Works

1. **API Health Check**
   - Verifies the API is running at `http://localhost:8000`
   - Ensures the service is ready before starting evaluation

2. **Evaluation Initiation**
   - Sends POST request to `/api/evaluate/golden-dataset`
   - Receives a task ID for tracking the evaluation

3. **Progress Monitoring**
   - Polls `/api/evaluate/status/{task_id}` every 2 seconds
   - Displays real-time progress updates:
     - Completed questions count
     - Failed questions count
     - Current set and question being processed
     - Completion percentage

4. **Completion Detection**
   - Monitors for "completed" status
   - Implements timeout protection (default: 30 minutes)
   - Tracks periods of inactivity to detect stalls

5. **Results Display**
   - Shows final statistics (passed/failed/total)
   - Provides hyperlinked path to results file
   - Uses terminal escape sequences for clickable file paths

### Script Execution Flow

```python
# 1. Check API availability
if not check_api_health(base_url):
    exit()

# 2. Start evaluation
response = requests.post(f"{base_url}/api/evaluate/golden-dataset")
task_id = response.json()["task_id"]

# 3. Monitor progress
while status != "completed":
    status_response = requests.get(f"{base_url}/api/evaluate/status/{task_id}")
    # Display progress updates
    # Check for timeout or stalls

# 4. Display results
print(f"Results file: {make_hyperlink(results_file)}")
```

### Key Features

- **Progress Visualization**: Shows which question/set is currently being processed
- **Timeout Protection**: Prevents infinite waiting with configurable timeout
- **Error Handling**: Gracefully handles API errors and connection issues
- **Hyperlinked Output**: Makes result files clickable in supported terminals
- **Activity Monitoring**: Detects evaluation stalls and provides warnings

## RAG Pipeline Integration

### Answer Generation Process

1. **Question Input**
   - Evaluation framework receives a question from the dataset
   - Question is passed to the RAG pipeline via `chat_with_content`

2. **RAG Pipeline Execution**
   - **Search Query Generation**: Creates optimized search query using HyDe
   - **Vector Search**: Queries Azure AI Search for relevant content
   - **Result Review**: AI reviews and filters search results
   - **Answer Synthesis**: Generates comprehensive answer from vetted results

3. **Response Extraction**
   - Evaluation runner extracts the 'answer' field from RAG response
   - Handles various response formats for compatibility

### Model Separation

The framework uses two distinct Azure OpenAI deployments:

1. **RAG Pipeline Model** (`AZURE_OPENAI_DEPLOYMENT_NAME`)
   - Used for generating answers from indexed content
   - Typically configured as gpt-4o or similar
   - Handles search query generation and answer synthesis

2. **Evaluation Model** (`AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME`)
   - Used exclusively for evaluating answer quality
   - Default: gpt-4.1
   - Compares generated answers against ground truth

This separation ensures:
- Evaluation independence from the model being tested
- Flexibility to use different models for different purposes
- Clear distinction between generation and evaluation

## Golden Dataset CSV Format

### Required Columns

1. **question** (required)
   - The question to be answered by the RAG pipeline
   - Should be clear and specific
   - Example: "What was the change in the Bain Consumer Health Index?"

2. **answer** or **expected_answer** (required)
   - The ground truth answer for comparison
   - Should contain key facts expected in the response
   - Example: "The index increased by 0.1 points to 97.7"

### Metadata Columns (Optional but Recommended)

3. **set_id**
   - Groups related questions together
   - Format: "Set_1", "Set_2", etc.
   - Used for organizing evaluation results

4. **question_id**
   - Unique identifier within a set
   - Format: "Q1", "Q2", etc.
   - Enables precise progress tracking

5. **content_type**
   - Type of source content: "document" or "web"
   - Helps categorize evaluation results

6. **source_files**
   - Comma-separated list of source file names
   - Documents where answers should be found
   - Example: "report.pdf,presentation.pptx"

7. **validation_notes**
   - Notes about what the question tests
   - Example: "Tests numerical data extraction"

### Additional Metadata Columns

8. **source_path** - Original file location
9. **author_name** - Content author
10. **author_email** - Author contact
11. **organization** - Source organization
12. **subject** - Email subject line
13. **received_date** - Content receipt date
14. **key_topics** - Comma-separated topics
15. **provenance_description** - Content origin details
16. **web_url** - URL for web content
17. **pages_total** - Total pages in document

### Example CSV Entry

```csv
set_id,question_id,question,answer,content_type,source_files,validation_notes
Set_1,Q1,"What was the Q4 revenue?","Q4 revenue was $2.3 billion",document,"quarterly_report.pdf","Tests financial data extraction"
```

## Evaluation Rating System

The framework uses a 3-tier rating system:

### 5 Stars (Excellent)
- Answer contains all key information from ground truth
- Information is accurate and complete
- May include additional relevant context

### 3 Stars (Acceptable)
- Answer contains most key information
- Minor omissions or slight inaccuracies
- Generally useful but not perfect

### 1 Star (Poor)
- Missing significant information
- Contains errors or inaccuracies
- Does not adequately answer the question

## Configuration

### Environment Variables

```bash
# Evaluation-specific settings
EVALUATION_MAX_CONCURRENT=5              # Max parallel evaluations
AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME=gpt-4.1  # Model for evaluation
EVALUATION_TEMPERATURE=0.0               # Temperature for evaluation model

# RAG pipeline settings (used for answer generation)
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1     # Model for RAG pipeline
NUM_SEARCH_RESULTS=5                    # Search results per query
```

### Temperature Configuration

The evaluation framework supports configurable temperature settings:

- **EVALUATION_TEMPERATURE**: Controls randomness in evaluation model responses
  - Default: 0.0 (deterministic, recommended for consistent evaluations)
  - Range: 0.0 to 2.0
  - Lower values (0.0-0.3): More consistent, deterministic responses
  - Higher values (0.7-2.0): More creative, varied responses

**Note**: Some models have fixed temperature requirements:
- o1 and o3 family models (o1-preview, o1-mini, o3, o3-mini, o3-pro) automatically use temperature=1.0
- The framework handles this automatically, overriding the configured temperature when necessary

### File Locations

- **Golden Dataset**: `archive-chat-agent-api/tests/data/golden_dataset_template.csv`
- **Results Output**: `archive-chat-agent-api/tests/validation_results/`
- **Evaluation Script**: `archive-chat-agent-api/run_evaluation.py`

## Best Practices

### 1. Dataset Design
- Include diverse question types (factual, analytical, multi-hop)
- Cover different content sources (documents, web pages)
- Test edge cases (no results, ambiguous queries)
- Balance question difficulty across the dataset

### 2. Ground Truth Answers
- Focus on key facts rather than exact wording
- Include multiple acceptable phrasings when relevant
- Be specific about numerical values and dates
- Avoid overly long ground truth answers

### 3. Evaluation Execution
- Run evaluations during low-traffic periods
- Monitor API logs for errors during evaluation
- Review failed evaluations for patterns
- Use consistent evaluation models for comparisons

### 4. Results Analysis
- Track evaluation scores over time
- Identify question types with low scores
- Review the "thoughts" field for insights
- Use results to improve RAG pipeline tuning

## Troubleshooting

### Common Issues

1. **"No answer generated" Results**
   - Check RAG pipeline is returning 'answer' field
   - Verify content is indexed for test questions
   - Review API logs for generation errors

2. **Evaluation Timeouts**
   - Increase timeout in monitoring script
   - Check API server resources
   - Consider reducing concurrent evaluations

3. **Low Evaluation Scores**
   - Review search results quality
   - Check if content contains expected answers
   - Verify ground truth accuracy
   - Consider adjusting search parameters

4. **Progress Not Updating**
   - Ensure evaluation service callbacks are working
   - Check for API communication errors
   - Verify task ID is being tracked correctly

5. **Temperature Errors with Certain Models**
   - o1 and o3 family models (o1-preview, o1-mini, o3, o3-mini, o3-pro) require temperature=1.0
   - The framework automatically adjusts temperature based on the model
   - If you encounter temperature errors with a new model, add it to the TEMPERATURE_RESTRICTED_MODELS dict

## Performance Considerations

### Concurrency Settings
- Default: 5 concurrent evaluations
- Increase for faster processing (if API can handle)
- Decrease if experiencing rate limits or errors

### Evaluation Duration
- Typical: 1-2 seconds per question
- Factors: Model latency, search complexity, content size
- Total time ≈ (questions / concurrency) × 2 seconds

### Resource Usage
- Each evaluation uses ~2-3 API calls
- Memory usage scales with concurrent evaluations
- Network bandwidth for result transfers

## Future Enhancements

### Planned Features
1. **Evaluation Metrics Dashboard**
   - Visual progress tracking
   - Historical performance trends
   - Comparative analysis tools

2. **Advanced Evaluation Criteria**
   - Citation accuracy checking
   - Response relevance scoring
   - Factual consistency validation

3. **Automated Test Generation**
   - Generate questions from indexed content
   - Create ground truth from verified sources
   - Expand test coverage automatically

4. **Integration with CI/CD**
   - Automated evaluation on deployments
   - Performance regression detection
   - Quality gates for releases