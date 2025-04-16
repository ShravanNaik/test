from crewai import Agent, Task, Crew, Process
from textwrap import dedent
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

class PDFEvaluationSystem:
    def __init__(self, input_context, prompt_template, final_output):
        self.input_context = input_context
        self.prompt_template = prompt_template
        self.final_output = final_output
        self.setup_crew()
        
    def setup_crew(self):
        # Create specialized agents
        self.fact_checker = Agent(
            role="Factuality Analyst",
            goal="Evaluate the factual accuracy of the output against the input context and prompt rules",
            backstory=dedent("""
                You are an expert in information extraction and fact verification. Your specialty is 
                comparing extracted data against source documents to determine factual accuracy.
                You have years of experience in document analysis and can detect even subtle 
                discrepancies between inputs and outputs.
            """),
            verbose=True,
            allow_delegation=True
        )
        
        self.rules_evaluator = Agent(
            role="Rules Compliance Officer",
            goal="Analyze how well the output adheres to the rules defined in the prompt template",
            backstory=dedent("""
                You are a specialist in prompt engineering and rule-based systems. You excel at
                breaking down prompt templates, identifying the explicit and implicit rules,
                and verifying that outputs comply with these specifications. You have a
                methodical approach to analyzing rule adherence.
            """),
            verbose=True,
            allow_delegation=True
        )
        
        self.summary_quality_assessor = Agent(
            role="Summary Quality Expert",
            goal="Evaluate the conciseness, coherence, and clarity of the generated output",
            backstory=dedent("""
                You have extensive experience in content analysis and summary evaluation. 
                Your expertise lies in assessing whether summaries capture the essential 
                information while remaining cohesive, clear, and appropriately sized. You 
                understand the balance between brevity and completeness.
            """),
            verbose=True,
            allow_delegation=True
        )
        
        self.metrics_calculator = Agent(
            role="Performance Metrics Analyst",
            goal="Calculate precision, recall, and F1 score for the fact extraction component",
            backstory=dedent("""
                With a background in machine learning evaluation and information retrieval metrics,
                you specialize in quantifying performance through standard metrics. You can
                systematically count true positives, false positives, and false negatives to
                derive precision, recall, and F1 scores for extraction tasks.
            """),
            verbose=True,
            allow_delegation=True
        )
        
        self.evaluation_director = Agent(
            role="Evaluation Director",
            goal="Compile and synthesize all evaluation results into a comprehensive assessment report",
            backstory=dedent("""
                As an expert in research evaluation and assessment frameworks, you excel at
                integrating multiple evaluation perspectives into coherent, actionable reports.
                You can identify patterns across different evaluation dimensions and provide
                holistic insights about system performance.
            """),
            verbose=True,
            allow_delegation=True
        )
        
        # Create tasks
        self.task_extract_rules = Task(
            description=dedent(f"""
                Extract and list all the explicit rules and requirements from the prompt template.
                Identify each rule that specifies what information should be extracted or how
                the output should be formatted.
                
                Prompt template:
                {self.prompt_template}
                
                Your output should be a numbered list of rules found in the template.
            """),
            agent=self.rules_evaluator,
            expected_output="A comprehensive list of all extraction and formatting rules from the prompt template"
        )
        
        self.task_identify_facts = Task(
            description=dedent(f"""
                Identify all potential extractable facts from the input context that should be 
                included according to the rules in the prompt template.
                
                Input context:
                {self.input_context}
                
                Your output should be a numbered list of all extractable facts found in the input context.
            """),
            agent=self.fact_checker,
            expected_output="A comprehensive list of all extractable facts from the input context"
        )
        
        self.task_evaluate_fact_presence = Task(
            description=dedent(f"""
                Evaluate which facts from the identified list are actually present in the final output.
                Mark each fact as present (1) or absent (0).
                
                Final output:
                {self.final_output}
                
                Your output should be a list showing which facts were successfully extracted.
            """),
            agent=self.fact_checker,
            expected_output="An analysis of which facts were successfully extracted into the final output"
        )
        
        self.task_identify_incorrect_facts = Task(
            description=dedent(f"""
                Identify any information in the final output that is either:
                1. Not supported by the input context (false information)
                2. Misrepresented or distorted from the input context
                
                Input context:
                {self.input_context}
                
                Final output:
                {self.final_output}
                
                Your output should be a list of any incorrect or unsupported facts in the final output.
            """),
            agent=self.fact_checker,
            expected_output="A list of any incorrect or unsupported facts in the final output"
        )
        
        self.task_calculate_metrics = Task(
            description=dedent("""
                Using the results from the previous tasks, calculate:
                1. Precision: (correctly extracted facts) / (all extracted facts)
                2. Recall: (correctly extracted facts) / (all extractable facts)
                3. F1 Score: 2 * (precision * recall) / (precision + recall)
                
                Show your calculations and provide the final metrics.
            """),
            agent=self.metrics_calculator,
            expected_output="Precision, recall, and F1 score calculations with explanations"
        )
        
        self.task_evaluate_rule_compliance = Task(
            description=dedent(f"""
                Evaluate how well the final output complies with each rule identified from the prompt template.
                For each rule, provide a compliance score (0-10) and a brief explanation.
                
                Rules (from previous task):
                [Rules extracted in task_extract_rules]
                
                Final output:
                {self.final_output}
                
                Your output should be a table showing rule compliance scores and explanations.
            """),
            agent=self.rules_evaluator,
            expected_output="A detailed evaluation of rule compliance for the final output"
        )
        
        self.task_evaluate_conciseness = Task(
            description=dedent(f"""
                Evaluate the conciseness of the final output. Consider:
                1. Does it fit the "single screen goal" effectively?
                2. Is it appropriately condensed without losing essential information?
                3. Are there unnecessary details or redundancies?
                
                Final output:
                {self.final_output}
                
                Provide a conciseness score (0-10) with justification.
            """),
            agent=self.summary_quality_assessor,
            expected_output="An assessment of the output's conciseness with score and explanation"
        )
        
        self.task_evaluate_coherence = Task(
            description=dedent(f"""
                Evaluate the coherence of the final output. Consider:
                1. Does the information flow logically?
                2. Are there appropriate transitions between sections?
                3. Is the structured template used effectively?
                4. Is the narrative cohesive and well-organized?
                
                Final output:
                {self.final_output}
                
                Provide a coherence score (0-10) with justification.
            """),
            agent=self.summary_quality_assessor,
            expected_output="An assessment of the output's coherence with score and explanation"
        )
        
        self.task_evaluate_clarity = Task(
            description=dedent(f"""
                Evaluate the clarity of the final output. Consider:
                1. Is the language clear and precise?
                2. Are complex concepts explained appropriately?
                3. Is the text free of ambiguity?
                4. Is the formatting enhancing readability?
                
                Final output:
                {self.final_output}
                
                Provide a clarity score (0-10) with justification.
            """),
            agent=self.summary_quality_assessor,
            expected_output="An assessment of the output's clarity with score and explanation"
        )
        
        self.task_compile_report = Task(
            description=dedent("""
                Compile a comprehensive evaluation report that synthesizes all the assessments:
                1. Factuality and Faithfulness Analysis
                   - Overall assessment of fact accuracy
                   - Precision, recall, and F1 scores
                   - Identified incorrect or missing facts
                   
                2. Rule Compliance Analysis
                   - Overall rule adherence assessment
                   - Strengths and weaknesses in following prompt rules
                   
                3. Summary Quality Analysis
                   - Conciseness assessment
                   - Coherence assessment
                   - Clarity assessment
                   - Overall summary quality score
                   
                4. Recommendations for Improvement
                   - Specific suggestions to improve each evaluation dimension
                
                Make the report structured, insightful, and actionable.
            """),
            agent=self.evaluation_director,
            expected_output="A comprehensive evaluation report with scores, analysis, and recommendations"
        )
        
        # Create the crew
        self.crew = Crew(
            agents=[
                self.fact_checker,
                self.rules_evaluator,
                self.summary_quality_assessor,
                self.metrics_calculator,
                self.evaluation_director
            ],
            tasks=[
                self.task_extract_rules,
                self.task_identify_facts,
                self.task_evaluate_fact_presence,
                self.task_identify_incorrect_facts,
                self.task_calculate_metrics,
                self.task_evaluate_rule_compliance,
                self.task_evaluate_conciseness,
                self.task_evaluate_coherence,
                self.task_evaluate_clarity,
                self.task_compile_report
            ],
            process=Process.sequential,
            verbose=2
        )
    
    def run_evaluation(self):
        """Run the evaluation crew and return the comprehensive assessment"""
        result = self.crew.kickoff()
        return result

# Usage example
if __name__ == "__main__":
    # Sample data (replace with actual data)
    input_context = """
    XYZ Corporation reported Q3 earnings of $1.2B, up 15% year-over-year.
    Market share increased to 23% from 19% last quarter.
    The company launched 3 new products: ProductA, ProductB, and ProductC.
    CEO Jane Smith announced an expansion into European markets planned for Q1 2024.
    Customer satisfaction scores reached 4.8/5, the highest in company history.
    """
    
    prompt_template = """
    Extract the following information:
    1. Financial performance metrics (revenue, growth)
    2. Market position data
    3. New product information
    4. Future business plans
    5. Customer experience metrics
    
    Format the output as a concise executive summary that fits on a single screen.
    Use bullet points for key metrics.
    Include all dollar values and percentages exactly as they appear in the context.
    """
    
    final_output = """
    Executive Summary: XYZ Corporation Q3 Performance
    
    Financial Highlights:
    • Revenue: $1.2B (↑15% YoY)
    
    Market Position:
    • Market share: 23% (↑4% from previous quarter)
    
    Product Development:
    • 3 new products launched: ProductA, ProductB, ProductC
    
    Strategic Initiatives:
    • European market expansion planned for Q1 2024 (announced by CEO Jane Smith)
    
    Customer Metrics:
    • Satisfaction score: 4.8/5 (company record)
    """
    
    # Create and run the evaluation
    evaluator = PDFEvaluationSystem(input_context, prompt_template, final_output)
    evaluation_report = evaluator.run_evaluation()
    print(evaluation_report)
