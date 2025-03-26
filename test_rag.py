import os
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from query_data import query_rag
import time

load_dotenv()
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}

The model has provided its chain-of-thought reasoning above. Based on this reasoning, does the actual response correctly match the expected response? 
(Answer with 'true' or 'false' only.)
"""

test_cases = [
    {
        "name": "test_monopoly_rules",
        "question": "How much total money does a player start with in Monopoly? (Answer with the number only)",
        "expected_response": "$1500"
    },
    {
        "name": "test_ticket_to_ride_rules",
        "question": "How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        "expected_response": "10 points"
    },
]

log_file = "log/rag_test_log.txt"
results = []

def generate_evaluation(prompt, model="gemini-2.0-flash", retries=3):
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=1,
        top_k=1,
        max_output_tokens=300,
        response_mime_type="text/plain",
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    
    response_text_eval = ""
    delay = 2
    for attempt in range(retries):
        try:
            for chunk in genai_client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text_eval += chunk.text.strip().lower()
            break  # break out if successful
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
    else:
        raise Exception("Max retries reached for evaluation generation.")
    
    return response_text_eval

def query_and_validate(question: str, expected_response: str):
    # Call the main query pipeline
    response_text = query_rag(question)

    # Format the evaluation prompt with chain-of-thought instruction
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    eval_result = generate_evaluation(prompt)
    
    return response_text.strip(), eval_result.strip()

def run_tests():
    correct = 0
    total = len(test_cases)
    logs = []

    for case in test_cases:
        name = case["name"]
        question = case["question"]
        expected = case["expected_response"]

        actual_response, eval_result = query_and_validate(question, expected)
        passed = "true" in eval_result

        if passed:
            correct += 1

        log_entry = {
            "test_name": name,
            "question": question,
            "expected": expected,
            "actual": actual_response,
            "evaluation": eval_result,
            "passed": passed
        }
        logs.append(log_entry)

        print(f"{name} → {'✅' if passed else '❌'}")
        print(f"Evaluation: {eval_result}")
        print("---")

    precision = correct / total if total > 0 else 0
    recall = correct / total if total > 0 else 0  # here binary classification makes these equal

    # Save results to log file
    with open(log_file, "w") as f:
        f.write(f"RAG Evaluation Log - {datetime.now()}\n\n")
        for log in logs:
            f.write(f"Test: {log['test_name']}\n")
            f.write(f"Question: {log['question']}\n")
            f.write(f"Expected: {log['expected']}\n")
            f.write(f"Actual: {log['actual']}\n")
            f.write(f"Evaluation Result: {log['evaluation']}\n")
            f.write(f"Passed: {log['passed']}\n")
            f.write("---\n")
        f.write(f"\nFinal Precision: {precision:.2f}\n")
        f.write(f"Final Recall: {recall:.2f}\n")

    print(f"\n✅ Finished testing. Precision = {precision:.2f}, Recall = {recall:.2f}")
    print(f"📝 Logged results to {log_file}")

if __name__ == "__main__":
    run_tests()
