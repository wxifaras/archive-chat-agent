"""
Evaluation prompts for assessing LLM response quality against ground truth.
"""

# Correctness prompt for evaluating AI answers against ground truth
CORRECTNESS_PROMPT = """You are an AI evaluator. 
The "correctness metric" is a measure of if the generated answer is correct based on the context provided. The generated answer will be in a JSON document in the following format:

{
    "Question": "What information is available about the project timeline?",
    "Answer": "The project timeline shows a completion date of Q4 2024"
}

You will need to compare the "Answer" with the ground truth answer, given in the supplied context.
You need to compare them and score the content between one to five stars using the following rating scale:
One star: The answer is incorrect
Three stars: The answer is partially correct, but could be missing some key context or nuance that makes it potentially misleading or incomplete compared with the context provided.
Five stars: The answer is correct and complete based on the context provided.

You must also provide your reasoning as to why the rating you selected was given.

This rating value should always be either 1, 3, or 5.

You will add your thoughts and rating for the answer and return the following JSON document:
{
    "question": "{question}",
    "ground_truth": "{ground_truth}",
    "answer": "{answer}",
    "thoughts": "{thoughts}",
    "stars": {rating}
}

question: Using the provided context of a ground truth question and answer, determine if the Answer is correct.
context: {{context}}
answer: {{answer}}
"""