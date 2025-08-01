# Archive Chat Agent API

A FastAPI-based intelligent document processing and chat system that enables uploading, indexing, and querying archived content using Azure AI services.

## Features

- **Document Upload & Processing**: Upload JSON documents with attachments for automated text extraction and indexing
- **Intelligent Search**: Azure AI Search integration with semantic search capabilities  
- **Chat Interface**: Conversational AI powered by Azure OpenAI for querying indexed content
- **Document Intelligence**: Automatic text extraction from various file formats using Azure Document Intelligence
- **Evaluation Framework**: Built-in evaluation system for testing chat responses and model performance
- **Semantic Chunking**: Advanced text chunking using semantic similarity for better search results

## Quick Start

### Prerequisites

- Python 3.11+
- Azure subscription with the following services configured:
  - Azure OpenAI
  - Azure AI Search
  - Azure Storage Account
  - Azure Document Intelligence
  - Azure Cosmos DB

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd archive-chat-agent/archive-chat-agent-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (see [Configuration](#configuration))

4. Run the API:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Configuration

Create a `.env` file with the following required variables:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_STORAGE_ARCHIVE_CONTAINER_NAME=your_container_name

# Azure Document Intelligence
AZURE_DOCUMENTINTELLIGENCE_ENDPOINT=your_doc_intel_endpoint
AZURE_DOCUMENTINTELLIGENCE_API_KEY=your_doc_intel_key

# Azure AI Search
AZURE_AI_SEARCH_SERVICE_ENDPOINT=your_search_endpoint
AZURE_AI_SEARCH_SERVICE_KEY=your_search_key
AZURE_AI_SEARCH_INDEX_NAME=your_index_name

# Azure Cosmos DB
COSMOS_ENDPOINT=your_cosmos_endpoint
COSMOS_DATABASE_NAME=your_database_name
COSMOS_CONTAINER_NAME=your_container_name

# Processing Settings
PAGE_OVERLAP=10
MAX_TOKENS=1000
NUM_SEARCH_RESULTS=5
USE_SEMANTIC_CHUNKING=false
```

## API Endpoints

### Content Management

#### Upload Content
```http
POST /api/upload_content
Content-Type: multipart/form-data

Parameters:
- document_id: string (optional - auto-generated if not provided)
- json_file: file (required - JSON document to process)
- attachments: file[] (optional - attachment files)
```

#### Process Existing Blobs
```http
POST /api/process_existing_blobs
```
Processes all existing files in Azure Storage that haven't been indexed yet.

### Chat Interface

#### Chat with Content
```http
POST /api/chat
Content-Type: multipart/form-data

Parameters:
- user_id: string (required)
- session_id: string (required) 
- message: string (required)
```

Returns structured chat response with citations and thought process.

### Evaluation System

#### Single Evaluation
```http
POST /api/evaluate/single
Content-Type: application/json

{
  "question": "Your question here",
  "expected_answer": "Expected response",
  "context": "Additional context"
}
```

#### Batch Evaluation
```http
POST /api/evaluate/upload-csv
Content-Type: multipart/form-data

Parameters:
- file: CSV file with evaluation data
```

#### Check Evaluation Status
```http
GET /api/evaluate/status/{task_id}
```

### Health Check
```http
GET /
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/swagger`

## Architecture

The API is built with:

- **FastAPI**: Modern Python web framework
- **Azure OpenAI**: LLM for chat and embeddings
- **Azure AI Search**: Vector and hybrid search capabilities
- **Azure Storage**: Document and blob storage
- **Azure Document Intelligence**: Text extraction from files
- **Azure Cosmos DB**: Chat history and metadata storage
- **LangChain**: Advanced text processing and chunking

## Key Components

- **Content Service**: Orchestrates document processing, chunking, and indexing
- **Chat Service**: Handles conversational AI with RAG (Retrieval Augmented Generation)
- **Evaluation Framework**: Automated testing and validation of model responses
- **Azure Integration**: Seamless integration with Azure AI services

## Development

### Running in Development Mode
```bash
python main.py
```

The server runs with auto-reload enabled on `http://localhost:8000`

### Project Structure
```
archive-chat-agent-api/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── core/                   # Configuration and settings
├── models/                 # Pydantic data models
├── routes/                 # API route definitions
├── services/               # Business logic and Azure integrations
└── prompts/               # LLM prompts and templates
```

## License

See [LICENSE](../LICENSE) file for details.
