from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Any
import pandas as pd
import re
from pydantic import BaseModel, Field

class DataPoint(BaseModel):
    name: str = Field(description="Name of the requested data point")
    found: bool = Field(description="Whether the data point was found in the output")
    correct: bool = Field(description="Whether the found data point is accurate")
    details: str = Field(description="Additional details or explanation")

class PerformanceMetrics(BaseModel):
    total_sections: int = Field(description="Total number of sections across all documents")
    approved_sections: int = Field(description="Number of sections that were approved")
    hit_rate: float = Field(description="Hit rate (approved sections / total sections)")
    section_details: List[DataPoint] = Field(description="Details about each section")

# Document Analyzer Agent
document_analyzer = Agent(
    role="Document Analyzer",
    goal="Analyze input context extracted from PDFs and identify all available sections",
    backstory="""You are an expert at analyzing document structures and content.
    Your job is to carefully review extracted PDF content and identify all distinct
    sections and data points available in the source material.""",
    allow_delegation=True,
    verbose=True
)

# Prompt Inspector Agent
prompt_inspector = Agent(
    role="Prompt Inspector",
    goal="Analyze prompt template and identify all requested data points",
    backstory="""You are specialized in understanding prompt engineering and
    instruction analysis. You carefully examine prompt templates to create
    clear verification checklists based on requested information.""",
    allow_delegation=True,
    verbose=True
)

# Output Validator Agent
output_validator = Agent(
    role="Output Validator",
    goal="Validate final output against source context and requirements",
    backstory="""You are a meticulous validator who ensures that outputs match
    requirements and source data. You cross-reference content to determine
    accuracy and completeness of extracted information.""",
    allow_delegation=True,
    verbose=True
)

# Performance Calculator Agent
performance_calculator = Agent(
    role="Performance Calculator",
    goal="Calculate performance metrics including hit rate",
    backstory="""You are a data analyst specialized in performance metrics
    calculation. You transform validation results into clear performance 
    indicators and provide actionable insights for improvement.""",
    allow_delegation=True,
    verbose=True
)

class PDFExtractionAssessmentCrew:
    def __init__(self, input_context: str, prompt_template: str, final_output: str):
        self.input_context = input_context
        self.prompt_template = prompt_template
        self.final_output = final_output
        
        # Define Tasks
        self.analyze_document_task = Task(
            description=f"""
            Analyze the provided input context extracted from PDFs.
            Identify all available sections and data points in the source material.
            Create a structured inventory of all data available in the source.
            
            Input Context: {input_context}
            
            Return your analysis as a structured list of all identified sections and data points.
            """,
            agent=document_analyzer,
            expected_output="A structured inventory of all sections and data points in the source material."
        )
        
        # This task will be created with document analysis results when run
        self.inspect_prompt_task = None
        self.validate_output_task = None
        self.calculate_performance_task = None
    
    def run_assessment(self) -> tuple:
        """Run the full assessment process and return performance metrics"""
        
        # Step 1: Create and run document analysis crew
        document_crew = Crew(
            agents=[document_analyzer],
            tasks=[self.analyze_document_task],
            verbose=True
        )
        document_analysis_output = document_crew.kickoff()
        # Access the actual result text
        document_analysis = document_analysis_output.raw_output
        
        # Step 2: Create and run prompt inspection crew
        self.inspect_prompt_task = Task(
            description=f"""
            Analyze the provided prompt template.
            Identify all 12 requested data points/posts specified in the prompt.
            Create a clear checklist of what should be present in the final output.
            
            Prompt Template: {self.prompt_template}
            
            Return your analysis as a structured checklist of all requested data points.
            """,
            agent=prompt_inspector,
            expected_output="A structured checklist of all 12 requested data points from the prompt template."
        )
        
        prompt_crew = Crew(
            agents=[prompt_inspector],
            tasks=[self.inspect_prompt_task],
            verbose=True
        )
        prompt_inspection_output = prompt_crew.kickoff()
        # Access the actual result text
        prompt_inspection = prompt_inspection_output.raw_output
        
        # Step 3: Create and run output validation crew
        self.validate_output_task = Task(
            description=f"""
            Cross-reference the final output against both the source context and prompt requirements.
            For each requested data point, determine if it is:
            - Found and correct
            - Found but incorrect
            - Missing
            - Not applicable
            
            Input Context: {self.input_context}
            Requested Data Points: {prompt_inspection}
            Final Output: {self.final_output}
            
            Return your validation results for each requested data point.
            """,
            agent=output_validator,
            expected_output="Validation results for each requested data point."
        )
        
        validation_crew = Crew(
            agents=[output_validator],
            tasks=[self.validate_output_task],
            verbose=True
        )
        validation_results_output = validation_crew.kickoff()
        # Access the actual result text
        validation_results = validation_results_output.raw_output
        
        # Step 4: Create and run performance calculation crew
        self.calculate_performance_task = Task(
            description=f"""
            Calculate performance metrics based on validation results:
            - Hit rate: number of approved sections / total sections
            - Additional relevant metrics
            
            Validation Results: {validation_results}
            Document Analysis: {document_analysis}
            
            Return a comprehensive performance report with metrics and suggestions for improvement.
            """,
            agent=performance_calculator,
            expected_output="A comprehensive performance report with metrics and suggestions for improvement."
        )
        
        performance_crew = Crew(
            agents=[performance_calculator],
            tasks=[self.calculate_performance_task],
            verbose=True
        )
        performance_report_output = performance_crew.kickoff()
        # Access the actual result text
        performance_report = performance_report_output.raw_output
        
        # Extract metrics from the performance report
        metrics = self._extract_metrics(performance_report, validation_results)
        
        return metrics, performance_report
    
    def _extract_metrics(self, performance_report: str, validation_results: str) -> PerformanceMetrics:
        """Extract structured metrics from the performance report"""
        
        # Parse validation results to get data points
        data_points = []
        if validation_results:
            validation_lines = validation_results.strip().split('\n')
            
            for line in validation_lines:
                if ':' in line and any(status in line.lower() for status in ['found', 'missing', 'correct', 'incorrect']):
                    name = line.split(':')[0].strip()
                    found = 'found' in line.lower() and 'not found' not in line.lower()
                    correct = 'correct' in line.lower() and 'incorrect' not in line.lower()
                    details = line.split(':', 1)[1].strip() if ':' in line else ""
                    
                    data_points.append(DataPoint(
                        name=name,
                        found=found,
                        correct=correct,
                        details=details
                    ))
        
        # Extract hit rate metrics
        hit_rate_pattern = r"Hit Rate:?\s*(\d+\.?\d*)%?"
        hit_rate_match = re.search(hit_rate_pattern, performance_report) if performance_report else None
        hit_rate = float(hit_rate_match.group(1))/100 if hit_rate_match else 0.0
        
        total_pattern = r"Total Sections:?\s*(\d+)"
        total_match = re.search(total_pattern, performance_report) if performance_report else None
        total_sections = int(total_match.group(1)) if total_match else len(data_points)
        
        approved_pattern = r"Approved Sections:?\s*(\d+)"
        approved_match = re.search(approved_pattern, performance_report) if performance_report else None
        approved_sections = int(approved_match.group(1)) if approved_match else sum(1 for dp in data_points if dp.found and dp.correct)
        
        return PerformanceMetrics(
            total_sections=total_sections,
            approved_sections=approved_sections,
            hit_rate=hit_rate,
            section_details=data_points
        )
    
    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a formatted report from performance metrics"""
        
        report = f"""
        # PDF Extraction Performance Assessment Report
        
        ## Summary Metrics
        - Total Sections: {metrics.total_sections}
        - Approved Sections: {metrics.approved_sections}
        - Hit Rate: {metrics.hit_rate:.2%}
        
        ## Section Details
        """
        
        for dp in metrics.section_details:
            status = "✅ Found & Correct" if dp.found and dp.correct else "❌ Found but Incorrect" if dp.found else "❓ Missing"
            report += f"\n- {dp.name}: {status}\n  {dp.details}"
        
        return report

# Example usage
def run_pdf_assessment(input_context, prompt_template, final_output):
    crew = PDFExtractionAssessmentCrew(
        input_context=input_context,
        prompt_template=prompt_template,
        final_output=final_output
    )
    
    metrics, performance_report = crew.run_assessment()
    report = crew.generate_report(metrics)
    
    return report, performance_report

# Example:
if __name__ == "__main__":
    # These would be your actual inputs
    input_context = """Sample PDF extracted content with multiple sections..."""
    prompt_template = """Template with 12 required data points..."""
    final_output = """Final output generated based on the prompt..."""
    
    report, raw_performance = run_pdf_assessment(input_context, prompt_template, final_output)
    print(report)
