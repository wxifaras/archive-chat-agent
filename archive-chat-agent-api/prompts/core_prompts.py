"""
This module contains the system prompts for the ARCHIVE Chat API. 
These prompts are used to guide the behavior of the chat model and 
provide context for its responses.
"""

PROVENANCE_SOURCE_SYSTEM_PROMPT = """
You are a data extraction assistant. You will be provided with text containing Provenance information.
Your task is to identify and extract the source of the Provenance from the given text.
"""