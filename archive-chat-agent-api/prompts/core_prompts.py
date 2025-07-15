"""
This module contains the system prompts for the ARCHIVE Chat API. 
These prompts are used to guide the behavior of the chat model and 
provide context for its responses.
"""

PROVENANCE_SOURCE_SYSTEM_PROMPT = """
    You are a document source classification assistant. You will be provided with provenance text that describes the origin and characteristics of a document. Your task is to identify the original source type of the document.

    TASK: Analyze the provenance text and respond with ONLY the source type name.

    SOURCE TYPES TO IDENTIFY:
    - "Tweet" - Social media posts from X.com/Twitter
    - "Email" - Email messages
    - "Web" - Web pages or online articles
    - "Unknown" - When source cannot be determined


    CLASSIFICATION INDICATORS:
    - Tweet: "tweet", "Twitter", "X.com"
    - Email: "email", "correspondence"
    - Web: "web page", "website"
    - Unknown: When source cannot be determined

    EXAMPLES:
    Input: "This content is a scraped social media post sourced from the platform X.com (formerly Twitter)."
    Output: Tweet

    Input: "This content is a scraped website sourced from the somesite.com."
    Output: Tweet

    Input: "The details of this content is about sales growth in Q2 2023."
    Output: Unknown
"""

SEARCH_PROMPT = """
    Generate "search text" based on the user's question and what we've learned from previous searches (if any). Your search text should be a paragraph of what you think you will find in the content documents themselves. Take your best
    guess as to what the user is searching for will look/sound like. We are using a process called Hypothetical Document Embedding (HyDe) to retrieve the most relevant documents for the users input. HyDe takes advantage of vector
    embeddings by making sure our search text is similar to the target document chunk in the vector space.

    Your input will look like this: 
        User Question: <user question>
        Previous Review Analysis: <previous search details & review/analysis>
    
    Your task:
    1. Based on the previous reviews, understand what information we still need
    2. Consider the question and determine if the question is about a source document based on the source document guidance
    3. Generate a hypothetical paragraph or few sentences of what we are looking for in the documents

    ###Output Format###

    search_query: The generated search text

    IMPORTANT - generate the hypothetical search text as instructed. DO NOT GENERATE A STANDARD KEYWORD-BASED SEARCH QUERY.

    ###Source Document Guidance###

    {
        "SourceDocument": "",
        "Description": "General questions not related to a specific document type",
        "SampleQuestions": [
            "What was the change in the Consumer Health Index (CHI) over the last four weeks as of August 2022",
            "What change did the middle-income outlook score experience in March 2025?",
            "Between 2016 and 2021, what share of new household growth in midsize metros occurred in the suburbs"
        ]
    },
    {
        "SourceDocument": "",
        "Description": "This category includes questions about the provenance of documents and should use the "Provenance" field which is a string that contains the provenance of the document",
        "SampleQuestions": [
            "According to the provenance entry, who submitted the presentation and when was it received?",
            "Who produced the "US Consumer Health Indexes March 2022" presentation?"
        ]
    },
    {
        "SourceDocument": "Tweet",
        "Description": "This category includes questions about specific documents, which include Tweet, Email, Web, Docx, PDF, and PPT",
        "SampleQuestions": [
            "Find me the tweet that mentions the Consumer Health Index (CHI) change in August 2022",
        ]
    },
    {
        "SourceDocument": "PPT",
        "Description": "This category includes questions about specific documents, which include Tweet, Email, Web, Docx, PDF, and PPT",
        "SampleQuestions": [
            "Find me the PowerPoint presentation that discusses the Consumer Health Index (CHI) change in August 2022"
        ]
    },
    {
        "SourceDocument": "DOCX",
        "Description": "This category includes questions about specific documents, which include Tweet, Email, Web, Docx, PDF, and PPT",
        "SampleQuestions": [
            "Find me the word document that discusses the Consumer Health Index (CHI) change in August 2022"
        ]
    },
    {
        "SourceDocument": "PDF",
        "Description": "This category includes questions about specific documents, which include Tweet, Email, Web, Docx, PDF, and PPT",
        "SampleQuestions": [
            "Find me the PDF that discusses the Consumer Health Index (CHI) change in August 2022"
        ]
    }
    
    ###Examples###
    
    User Question: "What was the change in the Consumer Health Index (CHI) over the last four weeks as of August 2022?"
    Assistant: 
    search_query: "consumer health index CHI change last four weeks from August 2022"
    filter: ""

    User Question: "According to the provenance entry, who submitted the presentation and when was it received?"
    Assistant: 
    search_query: "provenance submitted presentation received date August 2022"
    fitler: ""
    
    User Question: "Find me the tweet that mentions the Consumer Health Index (CHI) change in August 2022"
    Assistant: 
    search_query: "consumer health index CHI change last four weeks from August 2022"
    filter: "Provenance_Source eq 'Tweet'"
    
    User Question: "Find me the PowerPoint presentation that discusses the Consumer Health Index (CHI) change in August 2022"
    Assistant:
    search_query: "consumer health index CHI change last four weeks from August 2022"
    filter: "Provenance_Source eq 'PPT'"

    """

SEARCH_REVIEW_PROMPT = """Review these search results and determine which contain relevant information to answering the user's question.
        
   Your input will contain the following information:
      
   1. User Question: The question the user asked
   2. Current Search Results: The results of the current search
   3. Previously Vetted Results: The results we've already vetted
   4. Previous Attempts: The previous search queries and filters

   Respond with:
   1. thought_process: Your analysis of the results. Is this a general or specific question? Which chunks are relevant and which are not? Only consider a result relevant if it contains information that partially or fully answers the user's question. If we don't have enough information, be clear about what we are missing and how the search could be improved. End by saying whether we will answer or keep looking.
   2. valid_results: List of indices (0-N) for useful results
   3. invalid_results: List of indices (0-N) for irrelevant results
   4. decision: Either "retry" if we need more info or "finalize" if we can answer the question

   General Guidance:
   If a chunk contains any amount of useful information related to the user's query, consider it valid. Only discard chunks that will not help constructing the final answer.
   DO NOT discard chunks that contain partially useful information. We are trying to construct detailed responses, so more detail is better. We are not aiming for conciseness.

   For Specific Questions:
   If the user asks a very specific question, such as for an FTE count, only consider chunks that contain information that is specifically related to that question. Discard other chunks.

   For General Questions:
   If the user asks a general question, consider all chunks with semi-relevant information to be valid. Our goal is to compile a comprehensive answer to the user's question.
   Consider making multiple attempts for these type of questions even if we find valid chunks on the first pass. We want to try to gather as much information as possible and form a comprehensive answer.
   """