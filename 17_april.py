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
            role="Prompt Rules & Accuracy Specialist",
            goal="Extract exactly 10 content extraction rules and all formatting rules with maximum precision and structure",
            backstory="You specialize in systematic prompt analysis with exceptional attention to detail. Your expertise is breaking down complex prompts into clearly defined rule sets, particularly when they contain numbered requirements. You excel at identifying both the explicit content extraction requirements and the formatting specifications needed for perfect output compliance.",
            verbose=True,
            llm=self.llm
        )
        
        fact_verifier = Agent(
            role="Rule-Based Factual Verification Expert",
            goal="Systematically verify each of the 10 content rules with maximum precision and detailed evidence-based analysis",
            backstory="You are a meticulous fact-checker with specialized expertise in rule-based content verification. Your unique skill is methodically comparing structured outputs against source materials while maintaining perfect organization across multiple verification criteria. You approach verification by examining each rule individually, gathering evidence from the source text, and providing detailed explanations supported by direct quotes. You're particularly skilled at detecting subtle discrepancies, omissions, and misrepresentations that others might miss.",
            verbose=True,
            llm=self.llm
        )
        
        format_verifier = Agent(
            role="Format Compliance Expert",
            goal="Verify with maximum precision if the output follows all formatting and structural rules from the prompt template",
            backstory="You specialize in analyzing document structure and format requirements with exceptional attention to detail. You can detect even the most subtle deviations from requested formats, structures, and stylistic guidelines while ensuring complete adherence to specifications.",
            llm=self.llm
        )
        
        metrics_analyst = Agent(
            role="Metrics Analyst",
            goal="Calculate precision, recall, F1 score and rule compliance metrics with statistical rigor and comprehensive analysis",
            backstory="You're a senior data scientist specializing in evaluation metrics for NLP tasks with expertise in statistical analysis and performance measurement. You can quantify both fact extraction accuracy and rule compliance with exceptional precision, providing detailed confidence intervals and significance testing.",
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
            The prompt template contains EXACTLY 10 specific content extraction points (numbered 1-10) plus several markdown formatting requirements. Your task is to:
            
            1. Identify and extract each of the 10 content extraction points as separate content rules (C1-C10)
            2. Identify all formatting and structural requirements for the markdown output as separate format rules (F1, F2, etc.)
            
            Prompt Template:
            {self.prompt_template}
            
            SYSTEMATIC APPROACH:
            
            STEP 1: CONTENT RULES EXTRACTION
            - Look for the 10 numbered points in the prompt (they may be formatted as "1.", "1)", "Point 1:", etc.)
            - For each numbered point (1-10), create a separate content rule
            - Capture the EXACT requirement from each point, preserving any specific terminology
            - Each content rule should focus on ONE distinct type of information to extract
            
            STEP 2: FORMAT RULES EXTRACTION
            - Look for ALL instructions about how to structure or format the output
            - Include rules about markdown formatting (headers, lists, bold/italic text, etc.)
            - Include rules about organization, section order, or hierarchical structure
            - Include rules about styling, tone, length limitations, etc.
            - Include any rules about how specific types of content should be presented
            
            Your output MUST be a structured JSON with two sections:
            1. "content_rules": EXACTLY 10 rules (C1-C10) mapping to each numbered point
            2. "format_rules": All identified formatting/structural requirements
            
            For each rule, include:
            - "rule_id": A unique identifier (C1-C10 for content rules, F1, F2... for format rules)
            - "description": Clear description of the rule (use exact wording from prompt where possible)
            - "importance": "critical", "important", or "minor"
            - "source_text": The exact text from the prompt that this rule is derived from
            
            VERIFICATION CHECKLIST:
            - Have you identified EXACTLY 10 content rules (C1-C10)?
            - Does each content rule directly correspond to one of the 10 numbered points?
            - Have you captured ALL formatting requirements for the markdown output?
            - Have you preserved the exact terminology and requirements from the prompt?
            
            This task is CRITICAL for the accuracy of the entire verification system.
            """,
            agent=prompt_analyzer
        )
        
        task_verify_facts = Task(
        description=f"""
        Systematically verify if the facts in the final output accurately reflect the input context according to the 10 specific content rules (C1-C10).
        
        Input Context:
        {self.input_context[:10000]}... (if the context is longer, please analyze it entirely)
        
        Final Output:
        {self.final_output}
        
        Content Rules:
        {{task_extract_rules}}
        
        VERIFICATION PROCESS:
        
        STEP 1: RULE-BY-RULE VERIFICATION
        For EACH of the 10 content rules (C1-C10):
        1. Identify the specific content in the output that corresponds to this rule
        2. Find the corresponding information in the input context
        3. Verify if the extracted information is:
           - PRESENT (was the required information included?)
           - ACCURATE (does it match what's in the input context?)
           - COMPLETE (was all required information for this rule included?)
        
        STEP 2: FACT CATEGORIZATION
        Categorize each extracted piece of information as:
        1. TRUE POSITIVE: Information correctly extracted according to the rule
        2. FALSE POSITIVE: Information that is either:
           - Not present in the input context
           - Significantly distorted from what appears in the input context
           - Not relevant to the rule it claims to address
        3. FALSE NEGATIVE: Key information in the input context that should have been extracted according to the rule but was missed
        
        Format your response as a structured JSON with these sections:
        
        ```json
        {
          "rule_based_verification": {
            "C1": {
              "rule_description": "[Copy the rule description here]",
              "content_in_output": "[The content found in the output for this rule]",
              "expected_from_input": "[What should have been extracted from input]",
              "present": true/false,
              "accurate": true/false,
              "complete": true/false,
              "verification_notes": "[Detailed explanation of your findings]"
            },
            "C2": {
              // Same structure as C1
            },
            // ... Continue for C3-C10
          },
          "fact_categorization": {
            "true_positives": [
              {
                "rule_id": "C1",
                "fact_text": "[The correctly extracted fact]",
                "location_in_input": "[Where this appears in the input]",
                "verification": "[Why this is correct]"
              },
              // Additional true positives
            ],
            "false_positives": [
              {
                "rule_id": "C3",
                "incorrect_fact": "[The incorrect information in output]",
                "actual_input": "[What the input actually says, if anything]",
                "issue_type": "not_in_input|distortion|irrelevant",
                "explanation": "[Detailed explanation of the error]"
              },
              // Additional false positives
            ],
            "false_negatives": [
              {
                "rule_id": "C5",
                "missing_fact": "[Information that should have been included]",
                "location_in_input": "[Where this appears in the input]",
                "importance": "critical|important|minor",
                "explanation": "[Why this information should have been included]"
              },
              // Additional false negatives
            ]
          },
          "summary_statistics": {
            "rules_fully_satisfied": 0, // Count of rules with present=true, accurate=true, complete=true
            "rules_partially_satisfied": 0, // Count of rules with some issues
            "rules_not_satisfied": 0, // Count of rules with major issues
            "true_positives_count": 0,
            "false_positives_count": 0,
            "false_negatives_count": 0
          }
        }
        ```
        
        IMPORTANT VERIFICATION GUIDELINES:
        - Be extremely thorough in checking EACH of the 10 content rules
        - For each rule, explicitly state what was found in the output and what was expected based on the input
        - Pay special attention to facts that might be partially correct but missing important details
        - Note instances where the output contains speculative information not found in the input
        - If the input context doesn't contain information related to a specific rule, note this explicitly
        
        Your thoroughness in this verification is CRITICAL for an accurate assessment of the output quality.
        """,
        agent=fact_verifier,
        depends_on=[task_extract_rules]  # Or depends on task_validate_rules if you implemented that
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
