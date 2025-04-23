from crewai import Agent, Task, Crew, Process
from langchain.llms import OpenAI
import json
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import numpy as np

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
            goal="Systematically verify each of the 10 content rules with maximum precision and detailed evidence-based analysis including confusion matrix categorization",
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
            goal="Calculate confusion matrix metrics (TP, TN, FP, FN) and derive precision, recall, F1 score with statistical rigor",
            backstory="You're a senior data scientist specializing in evaluation metrics for NLP tasks with expertise in statistical analysis and performance measurement. You can quantify both fact extraction accuracy and rule compliance with exceptional precision, providing detailed confusion matrices and derived metrics.",
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
            - Include rules about how specific types of content should be presented
            
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
            3. Verify if the extracted information falls into one of these categories:
            
               - TRUE POSITIVE (TP): Facts found in input text that match the prompt template AND appear correctly in output
               - TRUE NEGATIVE (TN): Facts omitted from input text, yet labels preserved in output (showing as label with no value)
               - FALSE POSITIVE (FP): Facts in output that match prompt template but are hallucinations (not found in input text)
               - FALSE NEGATIVE (FN): Facts missing from output despite existing in input text and matching prompt template
            
            STEP 2: CONFUSION MATRIX CATEGORIZATION
            For each rule, categorize all relevant facts into these four categories with detailed evidence:
            
            Format your response as a structured JSON with these sections:
            
            ```json
            {
              "rule_based_verification": {
                "C1": {
                  "rule_description": "[Copy the rule description here]",
                  "content_in_output": "[The content found in the output for this rule]",
                  "corresponding_input": "[What was found in the input for this rule]",
                  "confusion_matrix": {
                    "true_positives": [
                      {
                        "fact": "[The correctly extracted fact]",
                        "input_location": "[Where this appears in the input]",
                        "output_location": "[Where this appears in the output]"
                      }
                    ],
                    "true_negatives": [
                      {
                        "label": "[The label preserved in output]",
                        "notes": "[Explanation of why this is correctly shown as empty/N/A]"
                      }
                    ],
                    "false_positives": [
                      {
                        "fact": "[The hallucinated fact]",
                        "output_location": "[Where this appears in the output]",
                        "evidence": "[Why this is considered a hallucination]"
                      }
                    ],
                    "false_negatives": [
                      {
                        "fact": "[The missing fact]",
                        "input_location": "[Where this appears in the input]",
                        "evidence": "[Why this should have been included]"
                      }
                    ]
                  },
                  "metrics": {
                    "tp_count": 0,
                    "tn_count": 0,
                    "fp_count": 0,
                    "fn_count": 0
                  }
                },
                "C2": {
                  // Same structure as C1
                },
                // ... Continue for C3-C10
              },
              "confusion_matrix_totals": {
                "total_tp": 0,
                "total_tn": 0,
                "total_fp": 0,
                "total_fn": 0
              },
              "primary_issues": {
                "hallucination_examples": [
                  // Most significant false positives
                ],
                "missing_fact_examples": [
                  // Most significant false negatives
                ]
              }
            }
            ```
            
            IMPORTANT VERIFICATION GUIDELINES:
            - Be extremely thorough in checking EACH of the 10 content rules
            - Remember the exact definitions:
              * TP = facts from input that matched prompt template and appear in output
              * TN = facts omitted from input text, yet labels preserved in final output
              * FP = facts in output that match prompt template but are hallucinations 
              * FN = facts missing from output despite existing in input text
            - For each fact, provide specific evidence from both input and output
            - Pay special attention to subtle differences that might change meaning
            
            Your thoroughness in this verification is CRITICAL for an accurate assessment of the output quality.
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
            Calculate comprehensive metrics based on the verification results with special focus on the confusion matrix.
            
            Factual Verification Results:
            {task_verify_facts}
            
            Format Compliance Results:
            {task_verify_format}
            
            STEP 1: EXTRACT CONFUSION MATRIX VALUES
            Extract the following values from the factual verification:
            - Total True Positives (TP): Sum of all true positives across all rules
            - Total True Negatives (TN): Sum of all true negatives across all rules
            - Total False Positives (FP): Sum of all false positives across all rules
            - Total False Negatives (FN): Sum of all false negatives across all rules
            
            STEP 2: CALCULATE CONFUSION MATRIX DERIVED METRICS
            Calculate:
            - Precision: TP / (TP + FP)
            - Recall: TP / (TP + FN)
            - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
            - Accuracy: (TP + TN) / (TP + TN + FP + FN)
            - Specificity: TN / (TN + FP)
            - False Positive Rate: FP / (FP + TN)
            - False Negative Rate: FN / (FN + TP)
            
            STEP 3: CALCULATE PER-RULE METRICS
            For each content rule (C1-C10):
            - Extract TP, TN, FP, FN counts
            - Calculate precision, recall, F1 score
            
            STEP 4: CALCULATE FORMAT COMPLIANCE METRICS
            - Format Rule Compliance: % of format rules followed correctly
            
            Format your response as a JSON object with clearly organized sections:
            ```json
            {
              "confusion_matrix": {
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "visualization_data": {
                  "matrix": [[TP, FP], [FN, TN]]
                }
              },
              "derived_metrics": {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "specificity": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0
              },
              "per_rule_metrics": {
                "C1": {
                  "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                  "precision": 0.0, "recall": 0.0, "f1": 0.0
                },
                // Repeat for C2-C10
              },
              "format_compliance": {
                "compliant_rules": 0,
                "total_rules": 0,
                "compliance_rate": 0.0
              }
            }
            ```
            
            Be precise in your calculations and ensure all metrics are between 0 and 1.
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
               - Overall assessment of output quality
               - Key confusion matrix metrics (TP, TN, FP, FN counts)
               - Precision, recall, F1 score
            
            2. Confusion Matrix Analysis
               - Detailed explanation of the confusion matrix results
               - Analysis of most common types of errors
               - Examples of significant hallucinations (FP)
               - Examples of important missed facts (FN)
            
            3. Per-Rule Performance
               - Rule-by-rule breakdown of performance
               - Identification of best and worst performing rules
               - Patterns in error types across rules
            
            4. Format Compliance Assessment
               - Overview of format rule compliance
               - Examples of format rule violations
            
            5. Recommendations for Improvement
               - Specific suggestions to reduce hallucinations (FP)
               - Specific suggestions to improve recall of important facts (reduce FN)
               - Format compliance improvements
            
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
    
    def visualize_results(self, metrics_json):
        """Generate visualizations for the verification results with focus on confusion matrix"""
        try:
            # Convert string to JSON if needed
            if isinstance(metrics_json, str):
                try:
                    metrics_json = json.loads(metrics_json)
                except:
                    # Try to extract JSON from text if direct parsing fails
                    json_pattern = r'\{[\s\S]*\}'
                    metrics_match = re.search(json_pattern, metrics_json)
                    if metrics_match:
                        metrics_json = json.loads(metrics_match.group(0))
                    else:
                        raise ValueError("Could not extract valid JSON from the results")
            
            # Create confusion matrix visualization
            if "confusion_matrix" in metrics_json:
                cm = metrics_json["confusion_matrix"]
                
                # Extract confusion matrix values
                tp = cm.get("true_positives", 0)
                tn = cm.get("true_negatives", 0)
                fp = cm.get("false_positives", 0)
                fn = cm.get("false_negatives", 0)
                
                # Create confusion matrix as 2x2 array
                conf_matrix = np.array([[tp, fp], [fn, tn]])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Predicted Positive', 'Predicted Negative'],
                           yticklabels=['Actual Positive', 'Actual Negative'])
                plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig("confusion_matrix.png")
                
                # Create horizontal bar chart for TP, TN, FP, FN counts
                plt.figure(figsize=(10, 6))
                metrics_counts = [tp, tn, fp, fn]
                categories = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
                colors = ['#4CAF50', '#2196F3', '#F44336', '#FF9800']  # Green, Blue, Red, Orange
                
                bars = plt.barh(categories, metrics_counts, color=colors)
                plt.xlabel('Count')
                plt.title('Confusion Matrix Counts')
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Add count labels to the bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                            f'{width}', ha='left', va='center')
                
                plt.tight_layout()
                plt.savefig("confusion_matrix_counts.png")
            
            # Derived metrics visualization
            if "derived_metrics" in metrics_json:
                derived = metrics_json["derived_metrics"]
                
                plt.figure(figsize=(10, 6))
                metrics_values = [
                    derived.get("precision", 0),
                    derived.get("recall", 0),
                    derived.get("f1_score", 0),
                    derived.get("accuracy", 0)
                ]
                metric_names = ["Precision", "Recall", "F1 Score", "Accuracy"]
                
                plt.bar(metric_names, metrics_values, color=['#3F51B5', '#009688', '#9C27B0', '#FF5722'])
                plt.title("Model Performance Metrics")
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add percentage labels on top of bars
                for i, v in enumerate(metrics_values):
                    plt.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig("derived_metrics.png")
            
            # Per-rule metrics visualization
            if "per_rule_metrics" in metrics_json:
                rules = metrics_json["per_rule_metrics"]
                rule_ids = list(rules.keys())
                
                if rule_ids:
                    # Create a dataframe for better visualization
                    rule_data = []
                    for rule_id in rule_ids:
                        rule_metrics = rules[rule_id]
                        rule_data.append({
                            'Rule': rule_id,
                            'Precision': rule_metrics.get('precision', 0),
                            'Recall': rule_metrics.get('recall', 0),
                            'F1': rule_metrics.get('f1', 0),
                            'TP': rule_metrics.get('tp', 0),
                            'TN': rule_metrics.get('tn', 0),
                            'FP': rule_metrics.get('fp', 0),
                            'FN': rule_metrics.get('fn', 0)
                        })
                    
                    df = pd.DataFrame(rule_data)
                    
                    # 1. Rule performance metrics
                    plt.figure(figsize=(12, 8))
                    melted_df = pd.melt(df, id_vars=["Rule"], 
                                       value_vars=["Precision", "Recall", "F1"],
                                       var_name="Metric", value_name="Value")
                    
                    sns.barplot(x="Rule", y="Value", hue="Metric", data=melted_df)
                    plt.title("Per-Rule Performance Metrics")
                    plt.xticks(rotation=45, ha="right")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.ylim(0, 1.0)
                    plt.tight_layout()
                    plt.savefig("per_rule_metrics.png")
                    
                    # 2. Rule confusion matrix counts
                    plt.figure(figsize=(14, 8))
                    cm_melted = pd.melt(df, id_vars=["Rule"], 
                                     value_vars=["TP", "TN", "FP", "FN"],
                                     var_name="Category", value_name="Count")
                    
                    sns.barplot(x="Rule", y="Count", hue="Category", data=cm_melted, 
                              palette={"TP": "#4CAF50", "TN": "#2196F3", "FP": "#F44336", "FN": "#FF9800"})
                    plt.title("Confusion Matrix Counts Per Rule")
                    plt.xticks(rotation=45, ha="right")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig("per_rule_confusion_matrix.png")
            
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
