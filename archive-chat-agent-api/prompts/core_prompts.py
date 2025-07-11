"""
This module contains the system prompts for the ARCHIVE Chat API. 
These prompts are used to guide the behavior of the chat model and 
provide context for its responses.
"""

PROVENANCE_SOURCE_SYSTEM_PROMPT = """
You are a data extraction assistant. 
Given text containing Provenance information, identify the source type: Tweets, Word (DOCX), PowerPoint (PPTX), or PDF.
Return only the source type.
"""