from crewai import Agent, Task, Crew, Process
from langchain.llms import OpenAI
import json
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

class ComprehensiveOutputVerifier:
    def __init__(self, input_context, prompt_template, final_output, llm_model="gpt-3.5-turbo"):
        """Initialize the comprehensive verification system.
        
        Args:
            input_context: The original text content extracted from the PDF
            prompt_template: The prompt template used to guide the extraction
            final_output: The generated output to verify
            llm_model: The language model to use for the CrewAI agents
        """
        self.input_context = input_context
        self.prompt_template = prompt_template
        self.final_output = final_output
        self.llm = OpenAI(model=llm_model, temperature=0.1)
        
    def create_crew(self):
        """Create and configure the CrewAI agents and tasks."""
        
        # Define the specialist agents
        prompt_analyzer = Agent(
            role="Prompt Rules Specialist",
            goal="Extract and interpret all rules from the prompt template",
            backstory="You specialize in understanding prompt engineering patterns and requirements. You can translate abstract prompt templates into concrete extraction and formatting rules.",
            verbose=True,
            llm=self.llm
        )
        
        fact_verifier = Agent(
            role="Factual Verification Expert",
            goal="Verify if facts in the output accurately reflect the input context",
            backstory="You're a meticulous fact-checker with years of experience verifying information accuracy. You compare output claims against source material to determine correctness.",
            verbose=True,
            llm=self.llm
        )
        
        format_verifier = Agent(
            role="Format Compliance Expert",
            goal="Verify if the output follows all formatting and structural rules from the prompt template",
            backstory="You specialize in analyzing document structure and format requirements. You can detect even subtle deviations from requested formats and structures.",
            verbose=True,
            llm=self.llm
        )
        
        metrics_analyst = Agent(
            role="Metrics Analyst",
            goal="Calculate precision, recall, F1 score and rule compliance metrics",
            backstory="You're a data scientist specializing in evaluation metrics for NLP tasks. You can quantify both fact extraction accuracy and rule compliance.",
            verbose=True,
            llm=self.llm
        )
        
        report_compiler = Agent(
            role="Assessment Report Compiler",
            goal="Compile a comprehensive assessment report with all findings and metrics",
            backstory="You excel at synthesizing complex analyses into clear, actionable reports. You highlight key findings and provide constructive feedback.",
            verbose=True,
            llm=self.llm
        )
        
        # Define the tasks
        task_extract_rules = Task(
            description=f"""
            Analyze the prompt template and extract TWO types of rules:
            1. Content extraction rules (what facts should be extracted)
            2. Format/structure rules (how the output should be formatted)
            
            Prompt Template:
            {self.prompt_template}
            
            Your output should be a structured JSON with two sections:
            1. "content_rules": List of rules about what facts should be extracted
            2. "format_rules": List of rules about output structure, formatting, etc.
            
            For each rule, include:
            - "rule_id": A unique identifier (C1, C2... for content rules, F1, F2... for format rules)
            - "description": Clear description of the rule
            - "importance": "critical", "important", or "minor"
            
            Be thorough and specific in identifying ALL rules in the prompt template.
            """,
            agent=prompt_analyzer
        )
        
        task_verify_facts = Task(
            description=f"""
            Verify if the facts in the final output accurately reflect the facts in the input context according to the content rules.
            
            Input Context:
            {self.input_context[:10000]}... (if the context is longer, please analyze it entirely)
            
            Final Output:
            {self.final_output}
            
            Content Rules:
            {{task_extract_rules}}
            
            For each fact/statement in the final output:
            1. Check if it exists in the input context
            2. Verify if it was extracted according to the relevant content rule
            3. Verify if it's accurately represented without distortion
            
            Create a detailed verification report with:
            1. True positives: Facts correctly extracted from the input context
            2. False positives: Facts in the output that are either not in the input context or significantly distorted
            3. False negatives: Key facts in the input context that should have been extracted according to the rules but were missed
            
            Format your response as a JSON object with these three categories.
            Each fact should include:
            - "fact_text": The text of the fact
            - "rule_id": Which rule it corresponds to
            - "verified": true/false (whether it accurately reflects the input context)
            - "explanation": Brief explanation of verification result
            """,
            agent=fact_verifier,
            depends_on=[task_extract_rules]
        )
        
        task_verify_format = Task(
            description=f"""
            Verify if the final output follows all formatting and structural rules from the prompt template.
            
            Final Output:
            {self.final_output}
            
            Format Rules:
            {{task_extract_rules}}
            
            For each format rule identified:
            1. Check if the final output complies with the rule
            2. Note any deviations or violations
            
            Create a detailed format compliance report with:
            1. Compliant rules: Format rules that were followed correctly
            2. Non-compliant rules: Format rules that were violated or not followed properly
            
            Format your response as a JSON object with these two categories.
            Each rule assessment should include:
            - "rule_id": The format rule identifier
            - "compliant": true/false
            - "explanation": Brief explanation of compliance or violation
            - "severity": "high", "medium", or "low" (impact of the violation)
            """,
            agent=format_verifier,
            depends_on=[task_extract_rules]
        )
        
        task_calculate_metrics = Task(
            description="""
            Calculate comprehensive metrics based on the verification results.
            
            Factual Verification Results:
            {task_verify_facts}
            
            Format Compliance Results:
            {task_verify_format}
            
            Calculate:
            
            1. Factual Accuracy Metrics:
               - Precision: (true positives) / (true positives + false positives)
               - Recall: (true positives) / (true positives + false negatives)
               - F1 Score: 2 * (precision * recall) / (precision + recall)
            
            2. Rule Compliance Metrics:
               - Content Rule Compliance: % of content rules followed correctly
               - Format Rule Compliance: % of format rules followed correctly
               - Overall Rule Compliance: % of all rules followed correctly
            
            3. Per-Rule Metrics:
               - Calculate precision and recall for each content rule
               - Calculate compliance rate for each format rule
            
            Format your response as a JSON object with clearly organized sections for each metric category.
            """,
            agent=metrics_analyst,
            depends_on=[task_verify_facts, task_verify_format]
        )
        
        task_compile_report = Task(
            description="""
            Compile a comprehensive assessment report with all findings and metrics.
            
            Extraction and Format Rules:
            {task_extract_rules}
            
            Factual Verification Results:
            {task_verify_facts}
            
            Format Compliance Results:
            {task_verify_format}
            
            Metrics:
            {task_calculate_metrics}
            
            Create a detailed report that includes:
            1. Executive Summary
            2. Factual Accuracy Assessment
               - Overall factual accuracy
               - Examples of correct and incorrect facts
            3. Rule Compliance Assessment
               - Content rule compliance
               - Format rule compliance
               - Examples of rule violations
            4. Detailed Metrics (overall and per-rule)
            5. Recommendations for Improvement
               - How to improve factual accuracy
               - How to improve rule compliance
            
            The report should be well-structured, clear, and actionable.
            """,
            agent=report_compiler,
            depends_on=[task_extract_rules, task_verify_facts, task_verify_format, task_calculate_metrics]
        )
        
        # Create the crew with sequential process
        crew = Crew(
            agents=[prompt_analyzer, fact_verifier, format_verifier, metrics_analyst, report_compiler],
            tasks=[task_extract_rules, task_verify_facts, task_verify_format, task_calculate_metrics, task_compile_report],
            verbose=2,
            process=Process.sequential
        )
        
        return crew
    
    def run_verification(self):
        """Run the verification process and return the results."""
        crew = self.create_crew()
        result = crew.kickoff()
        return result
    
    def visualize_results(self, results_json):
        """Generate visualizations for the verification results"""
        try:
            # Convert string to JSON if needed
            if isinstance(results_json, str):
                metrics_json = json.loads(results_json)
            else:
                metrics_json = results_json
            
            # Create factual accuracy metrics visualization
            plt.figure(figsize=(10, 6))
            factual_metrics = [
                metrics_json["factual_accuracy"]["precision"], 
                metrics_json["factual_accuracy"]["recall"], 
                metrics_json["factual_accuracy"]["f1_score"]
            ]
            
            plt.bar(["Precision", "Recall", "F1 Score"], factual_metrics, color=["blue", "green", "purple"])
            plt.title("Factual Accuracy Metrics")
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig("factual_accuracy_metrics.png")
            
            # Create rule compliance visualization
            plt.figure(figsize=(10, 6))
            compliance_metrics = [
                metrics_json["rule_compliance"]["content_rule_compliance"], 
                metrics_json["rule_compliance"]["format_rule_compliance"],
                metrics_json["rule_compliance"]["overall_rule_compliance"]
            ]
            
            plt.bar(["Content Rules", "Format Rules", "Overall"], compliance_metrics, color=["orange", "teal", "red"])
            plt.title("Rule Compliance Rates")
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig("rule_compliance_metrics.png")
            
            # Create per-rule metrics visualization if available
            if "per_rule_metrics" in metrics_json:
                # Content rules
                if "content_rules" in metrics_json["per_rule_metrics"]:
                    content_rules = metrics_json["per_rule_metrics"]["content_rules"]
                    rule_ids = list(content_rules.keys())
                    
                    if rule_ids:
                        precision_values = [content_rules[rule]["precision"] for rule in rule_ids]
                        recall_values = [content_rules[rule]["recall"] for rule in rule_ids]
                        
                        df = pd.DataFrame({
                            "Rule": rule_ids,
                            "Precision": precision_values,
                            "Recall": recall_values
                        })
                        
                        plt.figure(figsize=(12, 8))
                        melted_df = pd.melt(df, id_vars=["Rule"], 
                                           value_vars=["Precision", "Recall"],
                                           var_name="metric", value_name="value")
                        
                        sns.barplot(x="Rule", y="value", hue="metric", data=melted_df)
                        plt.title("Content Rule Performance Metrics")
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        plt.savefig("content_rule_metrics.png")
                
                # Format rules
                if "format_rules" in metrics_json["per_rule_metrics"]:
                    format_rules = metrics_json["per_rule_metrics"]["format_rules"]
                    format_rule_ids = list(format_rules.keys())
                    
                    if format_rule_ids:
                        compliance_values = [format_rules[rule]["compliance_rate"] for rule in format_rule_ids]
                        
                        plt.figure(figsize=(12, 6))
                        plt.bar(format_rule_ids, compliance_values, color="teal")
                        plt.title("Format Rule Compliance Rates")
                        plt.ylim(0, 1.0)
                        plt.xticks(rotation=45, ha="right")
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        plt.savefig("format_rule_compliance.png")
            
            return "Visualizations created successfully"
        
        except Exception as e:
            return f"Error creating visualizations: {str(e)}"

# Example usage
def main():
    # These would be your actual inputs
    input_context = """[Your PDF extracted content would go here]"""
    prompt_template = """[Your prompt template with extraction rules would go here]"""
    final_output = """[The generated output from your LLM would go here]"""
    
    verifier = ComprehensiveOutputVerifier(input_context, prompt_template, final_output)
    
    # Run verification
    verification_result = verifier.run_verification()
    
    # Print the full verification report
    print("\nFull Verification Report:")
    print(verification_result)
    
    # Try to extract metrics for visualization
    try:
        # Look for JSON-formatted metrics in the result
        json_pattern = r'\{[\s\S]*\}'
        metrics_match = re.search(json_pattern, verification_result)
        if metrics_match:
            metrics_json = json.loads(metrics_match.group(0))
            viz_result = verifier.visualize_results(metrics_json)
            print(viz_result)
    except Exception as e:
        print(f"Could not automatically generate visualizations: {str(e)}")

if __name__ == "__main__":
    main()
