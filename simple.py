from crewai import Agent, Task, Crew, Process
from textwrap import dedent

class FactualAccuracyEvaluator:
    def __init__(self, input_context, prompt_template, final_output):
        self.input_context = input_context
        self.prompt_template = prompt_template
        self.final_output = final_output
        self.setup_crew()
        
    def setup_crew(self):
        # Create a single agent focused on factual accuracy
        self.fact_checker = Agent(
            role="Factual Accuracy Evaluator",
            goal="Determine if the output accurately reflects facts from the input according to prompt rules",
            backstory=dedent("""
                You are an expert in information extraction and fact verification. Your specialty is 
                comparing extracted content against source documents to determine factual accuracy.
                You meticulously check if output facts match input facts according to the rules.
            """),
            verbose=True,
            allow_delegation=False
        )
        
        # Single task for factual accuracy evaluation
        self.task_evaluate_accuracy = Task(
            description=dedent(f"""
                Evaluate if the final output accurately reflects the facts in the input context
                according to the rules in the prompt template.
                
                Input context:
                {self.input_context}
                
                Prompt template:
                {self.prompt_template}
                
                Final output:
                {self.final_output}
                
                Focus only on factual accuracy. For each fact in the output:
                1. Identify if it exists in the input context
                2. Check if it was extracted according to the rules in the prompt
                3. Verify if it's represented accurately (numbers, names, details are correct)
                
                Provide a clear assessment of whether the output accurately reflects the 
                facts from the input according to the prompt template rules.
                
                Include specific examples of:
                - Correctly extracted facts
                - Any facts that are misrepresented or inaccurate
                - Any important facts that should have been extracted but weren't
            """),
            agent=self.fact_checker,
            expected_output="A detailed assessment of the factual accuracy of the output"
        )
        
        # Create the crew with just one agent and one task
        self.crew = Crew(
            agents=[self.fact_checker],
            tasks=[self.task_evaluate_accuracy],
            process=Process.sequential,
            verbose=2
        )
    
    def evaluate_accuracy(self):
        """Run the factual accuracy evaluation"""
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
    evaluator = FactualAccuracyEvaluator(input_context, prompt_template, final_output)
    accuracy_assessment = evaluator.evaluate_accuracy()
    print(accuracy_assessment)
