from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

# Your data
input_pdf_context = "The full text extracted from your PDF..."
input_prompt = "Summarize this PDF in one page focusing on key findings..."
output_summary = "The generated 1-page summary..."

# Create test case (expected_output is optional, so omit it)
test_case = LLMTestCase(
    input=input_prompt,
    actual_output=output_summary,
    context=input_pdf_context  # The source truth to compare against
)

# Define correctness metric (focus on factual consistency with context)
correctness_metric = GEval(
    name="Factual Consistency",
    evaluation_steps=[
        "Check if the 'actual_output' accurately reflects facts from the 'context' (PDF content)",
        "Penalize any claims in 'actual_output' that contradict or are unsupported by 'context'",
        "Penalize major omissions of key details from 'context'",
        "Allow paraphrasing but penalize inaccuracies",
        "Ignore stylistic differences but flag factual errors",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    # NOTE: We're using CONTEXT instead of EXPECTED_OUTPUT
)

# Run evaluation
evaluate([test_case], [correctness_metric])

# Print results
print("Score:", correctness_metric.score)  # 0.0 to 1.0 (1.0 = fully correct)
print("Reason:", correctness_metric.reason)  # Explanation of scoring
