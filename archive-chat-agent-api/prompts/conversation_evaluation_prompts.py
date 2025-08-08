"""
Evaluation prompts for multi-turn conversation assessment.
"""

# Prompt for evaluating individual turns within conversation context
CONVERSATION_TURN_EVALUATION_PROMPT = """You are evaluating a response within an ongoing multi-turn conversation.

Previous Conversation Context:
{conversation_history}

Current Turn:
Question: {current_question}
Expected Answer: {expected_answer}
Generated Answer: {generated_answer}

Please evaluate the generated answer considering:

1. **Correctness**: How accurate is the answer compared to the expected response?
2. **Context Usage**: Does the answer appropriately use information from previous turns?
3. **Consistency**: Is the answer consistent with information provided earlier?
4. **Conversation Flow**: Does the answer maintain natural conversation flow?

Rating Scale:
- 1 star: Incorrect answer or completely ignores relevant context
- 3 stars: Partially correct with adequate context usage but missing key elements
- 5 stars: Correct answer with excellent context usage and natural flow

You must return a JSON response in exactly this format:
{{
    "rating": <1, 3, or 5>,
    "correctness_assessment": "<detailed assessment of answer correctness>",
    "context_usage_assessment": "<how well the answer uses conversation context>",
    "evaluation_thoughts": "<overall evaluation with specific examples>"
}}"""


# Prompt for evaluating complete conversations holistically
CONVERSATION_HOLISTIC_EVALUATION_PROMPT = """You are evaluating an entire multi-turn conversation between a user and an AI assistant.

Complete Conversation:
{full_conversation}

Please evaluate the overall conversation quality considering:

1. **Context Coherence**: How well does the assistant maintain context throughout the conversation?
2. **Follow-up Handling**: Are follow-up questions addressed appropriately with reference to earlier context?
3. **Information Consistency**: Is information provided consistent across all turns?
4. **Conversation Flow**: Does the dialogue flow naturally from turn to turn?
5. **Overall Effectiveness**: Did the conversation successfully address the user's needs?

For each aspect, provide a score from 0.0 to 1.0 and specific examples.

Rating Scale for Overall Rating:
- 1 star: Poor conversation with significant context loss or inconsistencies
- 3 stars: Adequate conversation with some context usage but room for improvement
- 5 stars: Excellent conversation with strong context retention and natural flow

You must return a JSON response in exactly this format:
{{
    "overall_rating": <1, 3, or 5>,
    "context_coherence_score": <0.0 to 1.0>,
    "follow_up_accuracy": <0.0 to 1.0>,
    "information_consistency": <0.0 to 1.0>,
    "conversation_flow_score": <0.0 to 1.0>,
    "overall_effectiveness": <0.0 to 1.0>,
    "overall_evaluation_thoughts": "<detailed analysis of the conversation>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "specific_examples": {{
        "good_context_usage": ["<example 1>", ...],
        "missed_opportunities": ["<example 1>", ...],
        "inconsistencies": ["<example 1>", ...]
    }}
}}"""


# Prompt for evaluating context-dependent questions
CONTEXT_DEPENDENT_EVALUATION_PROMPT = """You are evaluating how well an AI assistant handles context-dependent questions in a conversation.

Previous Context:
{previous_context}

Context-Dependent Question: {question}
Expected Behavior: The answer should {expected_behavior}
Generated Answer: {generated_answer}

Evaluate specifically:
1. Did the assistant recognize this was a follow-up/clarification question?
2. Did it correctly reference the relevant previous context?
3. Did it provide the expected information based on context?

You must return a JSON response:
{{
    "context_recognized": <true/false>,
    "correct_reference": <true/false>,
    "evaluation": "<detailed evaluation>",
    "rating": <1, 3, or 5>
}}"""


# Helper prompt for identifying conversation patterns
CONVERSATION_PATTERN_ANALYSIS_PROMPT = """Analyze this conversation for common patterns and behaviors.

Conversation:
{conversation}

Identify:
1. Types of questions asked (factual, follow-up, clarification, etc.)
2. How well the assistant handles topic transitions
3. Any patterns in context usage or memory
4. Consistency of information across turns

Return a structured analysis that can be used to improve the system."""