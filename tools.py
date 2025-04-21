# Import required libraries
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, validator
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from rouge import Rouge
import re

# Define the Rouge Evaluator base class as provided
class RougeEvaluator(ABC, BaseModel):
    ground_truth: str
    predicted: str
    delimiter: str = " "
    
    @validator("ground_truth", pre=True)
    @classmethod
    def check_ground_truth(cls, value):
        if value is None:
            raise ValueError("Ground truth is a mandatory input and cannot be null.")
        return value
    
    @validator("predicted")
    @classmethod
    def check_predicted(cls, value):
        if value is None:
            raise ValueError("Predicted is a mandatory input and cannot be null.")
        return value
    
    @abstractmethod
    def evaluate(self):
        pass

# Implement the Rouge Metrics class as provided
class RougeMetrics(RougeEvaluator):
    def evaluate(self):
        rouge = Rouge()
        scores = rouge.get_scores(hyps=self.predicted, refs=self.ground_truth)
        all_metrics = scores[0]
        return all_metrics
    
    def __call__(self, *args, **kwargs):
        metrics = self.evaluate()
        return metrics

# Create a tool for Rouge evaluation
class RougeEvaluationTool(BaseTool):
    name: str = "Rouge Evaluation Tool"
    description: str = "Evaluates the similarity between predicted text and ground truth using Rouge metrics."
    
    def _run(self, ground_truth: str, predicted: str) -> Dict[str, Any]:
        evaluator = RougeMetrics(ground_truth=ground_truth, predicted=predicted)
        return evaluator()
    
    def _arun(self, ground_truth: str, predicted: str) -> Dict[str, Any]:
        # For async execution (falls back to sync in this implementation)
        return self._run(ground_truth, predicted)

# Tool to extract information based on prompt template
class InformationExtractionTool(BaseTool):
    name: str = "Information Extraction Tool"
    description: str = "Extracts key information points from text based on a prompt template."
    
    def _run(self, text: str, prompt_template: str) -> List[str]:
        # Basic implementation - in real scenarios, this would be more sophisticated
        # Extract key phrases from the prompt template
        key_phrases = re.findall(r'\{([^}]+)\}', prompt_template)
        
        # For each key phrase, try to find relevant information in the text
        extracted_info = []
        for phrase in key_phrases:
            # Simple regex-based extraction (would be more advanced in practice)
            pattern = f".*?({phrase.replace('_', ' ')}.*?)[.!?]"
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted_info.append(f"{phrase}: {matches[0].strip()}")
            else:
                extracted_info.append(f"{phrase}: No information found")
        
        return extracted_info
    
    def _arun(self, text: str, prompt_template: str) -> List[str]:
        return self._run(text, prompt_template)

# Define CrewAI agents
def create_text_validation_crew(input_text, prompt_template, generated_output):
    # Tools
    rouge_tool = RougeEvaluationTool()
    extraction_tool = InformationExtractionTool()
    
    # Define the Information Extractor Agent
    extractor_agent = Agent(
        role="Information Extractor",
        goal="Extract key information from text based on prompt templates",
        backstory=(
            "You are an expert in natural language processing with a specialization "
            "in information extraction. Your job is to identify and extract specific "
            "information from text based on prompt templates."
        ),
        tools=[extraction_tool],
        verbose=True
    )
    
    # Define the Validation Agent
    validation_agent = Agent(
        role="Content Validator",
        goal="Validate extracted information against original sources",
        backstory=(
            "You are a meticulous fact-checker with years of experience in validating "
            "content. You ensure that information extracted from sources accurately "
            "represents the original material."
        ),
        tools=[],
        verbose=True
    )
    
    # Define the Rouge Score Evaluator Agent
    evaluator_agent = Agent(
        role="Rouge Score Evaluator",
        goal="Calculate and analyze Rouge scores between text pairs",
        backstory=(
            "You are a metrics specialist focused on natural language evaluation. "
            "You use Rouge scores to quantify the similarity between texts and "
            "provide insights on content generation quality."
        ),
        tools=[rouge_tool],
        verbose=True
    )
    
    # Define tasks
    extraction_task = Task(
        description=(
            f"Analyze the prompt template: '{prompt_template}' and identify what "
            f"information it's designed to extract. Then, check the generated output: "
            f"'{generated_output}' and list out what specific information points were "
            f"actually extracted."
        ),
        agent=extractor_agent
    )
    
    validation_task = Task(
        description=(
            f"For each information point identified in the previous task, find the "
            f"corresponding information in the original text: '{input_text}'. "
            f"Create pairs of (extracted_info, original_info) for each point."
        ),
        agent=validation_agent,
        dependencies=[extraction_task]
    )
    
    evaluation_task = Task(
        description=(
            "For each pair of (extracted_info, original_info) from the previous task, "
            "calculate Rouge scores using the Rouge Evaluation Tool. Provide a "
            "detailed report of the scores and an overall assessment of the "
            "information extraction quality."
        ),
        agent=evaluator_agent,
        dependencies=[validation_task]
    )
    
    # Create the crew
    crew = Crew(
        agents=[extractor_agent, validation_agent, evaluator_agent],
        tasks=[extraction_task, validation_task, evaluation_task],
        verbose=True
    )
    
    return crew

# Function to run the validation process
def run_validation(input_text, prompt_template, generated_output):
    crew = create_text_validation_crew(input_text, prompt_template, generated_output)
    result = crew.kickoff()
    return result

# Example usage (uncomment and provide actual values to run)
"""
input_text = "Your original text source here..."
prompt_template = "Template with {placeholder} for information extraction"
generated_output = "The output generated from the template"

validation_result = run_validation(input_text, prompt_template, generated_output)
print(validation_result)
"""
