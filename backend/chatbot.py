from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import traceback
import random
import json
import os
import re


# --- CONFIGURATION ---
CSV_FILES = {
    'branch': 'data/global_ime_enhanced_branch_data.csv',
    'performance': 'data/global_ime_monthly_performance_enhanced.csv',
    'product_performance': 'data/global_ime_product_performance.csv',
    'customer_segments': 'data/global_ime_customer_segments.csv',
    'monthly_product': 'data/global_ime_monthly_product_data.csv',
}

load_dotenv()

# --- LOAD DATA ---
dataframes = {}
for name, path in CSV_FILES.items():
    if os.path.exists(path):
        dataframes[name] = pd.read_csv(path)
        print(f"✓ Loaded {name}: {path} ({len(dataframes[name])} rows)")
    else:
        print(f"✗ Warning: {path} not found. Skipping.")

print(f"\nLoaded {len(dataframes)} CSV files successfully!")


# Merge all three DataFrames on 'branch_name'
if os.path.exists("./data/global_ime_combined.csv"):
    merged_df = pd.read_csv("./data/global_ime_combined.csv")
else:
    merged_df = dataframes['branch'].merge(dataframes['performance'], on='branch_id', how='inner')
    merged_df = merged_df.merge(dataframes['customer_segments'], on='branch_id', how='inner')
    merged_df.to_csv("./data/global_ime_combined.csv", index=False)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def preprocess_data(df):
    df = df.copy()
    df['month_dt'] = pd.to_datetime(df['month'], format='%Y-%m')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'month']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    df = df.sort_values(['branch_id', 'month_dt'])
    return df, label_encoders

set_seeds(42)

# Preprocess data
df_processed, encoders = preprocess_data(merged_df)



# --- SETUP GEMINI ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
else:
    model_gemini = None
    print("Warning: GEMINI_API_KEY not set. LLM features will be disabled.")

def format_currency_npr(amount):
    """Format amount in NPR currency"""
    if pd.isna(amount) or amount == 0:
        return "NPR 0"
    return f"NPR {amount:,.2f}"

def get_dataset_info():
    """Get comprehensive information about all datasets"""
    dataset_info = {}
    for name, df in dataframes.items():
        dataset_info[name] = {
            'columns': list(df.columns),
            'row_count': len(df),
            'dtypes': df.dtypes.to_dict(),
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else [],
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'null_counts': df.isnull().sum().to_dict()
        }
    return dataset_info

def generate_dynamic_code(question, dataset_info):
    """Generate Python code dynamically using LLM to answer the question"""
    if not model_gemini:
        return "LLM not available. Cannot generate dynamic code."
    # Create a comprehensive prompt for code generation
    prompt_template = """
You are an expert data analyst and Python programmer. Generate Python code to answer the user's question about banking data.

USER QUESTION: "{question}"

AVAILABLE DATASETS:
{dataset_info_json}

IMPORTANT REQUIREMENTS:
1. All currency amounts are already in NPR (Nepalese Rupees) - no conversion needed
2. Use proper currency formatting: "NPR X,XXX.XX" for display
3. The dataframes are available as: dataframes['dataset_name']
4. Generate ONLY executable Python code, no explanations
5. The code should create a dictionary variable named `results`.
6. Handle missing data gracefully
7. Use pandas for data manipulation
8. Include proper error handling
9. Format currency amounts using the format_currency_npr() function
10. Make the code efficient and readable

EXAMPLE CODE STRUCTURE:
```python
try:
    # Your analysis code here
    results = {{}}
    
    # Example: Get revenue by branch
    if 'branch' in dataframes:
        df = dataframes['branch']
        if 'total_revenue' in df.columns and 'branch_name' in df.columns:
            branch_revenue = df.groupby('branch_name')['total_revenue'].sum()
            results['revenue_by_branch'] = {{
                branch: format_currency_npr(amount) 
                for branch, amount in branch_revenue.items()
            }}
    
    # The 'results' variable will be extracted and used. DO NOT use a return statement.
except Exception as e:
    results = {{'error': str(e)}}
```

Generate ONLY the Python code that will answer the question. The generated code MUST define a 'results' variable. Do not include a 'return' statement.
"""
    dataset_info_json = json.dumps(dataset_info, indent=2, default=str)
    code_generation_prompt = prompt_template.format(
        question=question,
        dataset_info_json=dataset_info_json
    ).replace('{{', '{').replace('}}', '}')

    try:
        response = model_gemini.generate_content(code_generation_prompt)
        code_text = response.text.strip()
        
        # Extract code from markdown if present
        if '```python' in code_text:
            code_match = re.search(r'```python\s*(.*?)\s*```', code_text, re.DOTALL)
            if code_match:
                code_text = code_match.group(1).strip()
        elif '```' in code_text:
            code_match = re.search(r'```\s*(.*?)\s*```', code_text, re.DOTALL)
            if code_match:
                code_text = code_match.group(1).strip()
        
        return code_text
        
    except Exception as e:
        return f"Error generating code: {str(e)}"

def execute_generated_code(code, question):
    """Execute the dynamically generated code safely"""
    try:
        # Create a safe execution environment
        local_vars = {
            'dataframes': dataframes,
            'pd': pd,
            'np': np,
            'format_currency_npr': format_currency_npr,
            'json': json
        }
        
        # Execute the generated code
        exec(code, globals(), local_vars)
        
        # Get the results
        if 'results' in local_vars:
            return local_vars['results']
        else:
            # Add the generated code to the error message for debugging
            return {'error': f'No results variable found in generated code. Code was:\\n{code}'}
            
    except Exception as e:
        error_msg = f"Error executing generated code: {str(e)}\nTraceback: {traceback.format_exc()}"
        return {'error': error_msg}

def generate_natural_response(question, results, dataset_info):
    """Generate a natural language response from the results"""
    if not model_gemini:
        return format_results_manually(results)
    
    response_prompt = f"""
You are a banking business analyst. Provide a clear, professional answer to the user's question based on the analysis results.

USER QUESTION: "{question}"

ANALYSIS RESULTS:
{json.dumps(results, indent=2, default=str)}

AVAILABLE DATASETS INFO:
{json.dumps(dataset_info, indent=2, default=str)}

INSTRUCTIONS:
1. Provide a direct answer to the question
2. Use the exact numbers from the results
3. Format currency as "NPR X,XXX.XX"
4. Be specific and professional
5. If there are multiple branches, list them clearly
6. If there's an error, explain it simply
7. Don't use technical jargon unless necessary
8. Focus on business insights
9. All amounts are already in NPR (Nepalese Rupees)
10. Add a short summary of the results

Respond with a clear, professional answer. Don't use JSON formatting or code blocks.
"""

    try:
        response = model_gemini.generate_content(response_prompt)
        return response.text.strip()
    except Exception as e:
        return format_results_manually(results)

def format_results_manually(results):
    """Manual formatting of results when LLM is not available"""
    if 'error' in results:
        return f"Error: {results['error']}"
    
    response_parts = []
    
    for key, value in results.items():
        if isinstance(value, dict):
            response_parts.append(f"\n{key.replace('_', ' ').title()}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    if any(keyword in key.lower() for keyword in ['revenue', 'income', 'profit', 'amount']):
                        response_parts.append(f"  • {sub_key}: {format_currency_npr(sub_value)}")
                    else:
                        response_parts.append(f"  • {sub_key}: {sub_value:,}")
                else:
                    response_parts.append(f"  • {sub_key}: {sub_value}")
        elif isinstance(value, (int, float)):
            if any(keyword in key.lower() for keyword in ['revenue', 'income', 'profit', 'amount']):
                response_parts.append(f"{key.replace('_', ' ').title()}: {format_currency_npr(value)}")
            else:
                response_parts.append(f"{key.replace('_', ' ').title()}: {value:,}")
        else:
            response_parts.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(response_parts) if response_parts else "No results found."

def intelligent_chatbot_response(question):
    """Main function that uses dynamic code generation to answer questions"""
    try:
        print("🔍 Analyzing question and generating code...")
        
        # Get dataset information
        dataset_info = get_dataset_info()
        
        # Generate dynamic code
        generated_code = generate_dynamic_code(question, dataset_info)
        
        if generated_code.startswith("Error"):
            return generated_code
        
        print("🧮 Executing generated code...")
        
        # Execute the generated code
        results = execute_generated_code(generated_code, question)
        
        if 'error' in results:
            return f"Error in analysis: {results['error']}"
        
        print("💡 Generating natural language response...")
        
        # Generate natural language response
        final_response = generate_natural_response(question, results, dataset_info)
        
        return final_response
        
    except Exception as e:
        return f"Error in intelligent analysis: {str(e)}"

def data_analysis_chat(question: str, chatHistory=None):
    """Main chat loop"""
    for name, df in dataframes.items():
        print(f"• {name}: {len(df)} records")

    print("=" * 60)
    
    try:
            
        if question.lower() in ['quit', 'exit', 'bye']:
            return "👋 Thank you for using the Dynamic Banking Data Chatbot!"
            
            
        print("\n🤖 Bot: Processing your question...")
        response = intelligent_chatbot_response(question)
        return response

    except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return "An error occurred while processing your question. Please try again or contact support if the issue persists."


# --- INSIGHTS GENERATION USING GEMINI ---
# This function generates insights based on historical data and future predictions using the Gemini LLM.

def generate_insights(months_hist, actual_hist, future_months, future_preds):
    """
    Generate insights from historical data and future predictions using Gemini LLM
    
    Args:
        months_hist: Historical months
        actual_hist: Historical actual values
        future_months: Future months for prediction
        future_preds: Predicted values for future months
    
    Returns:
        str: Generated insights about the data and predictions
    """
    try:        

        dataset_info = get_dataset_info()
        
        
        # Prepare data summary for Gemini
        data_summary = {
            'historical_data': list(zip(months_hist, actual_hist)),
            'future_predictions': list(zip(future_months, future_preds)),
            'historical_stats': {
                'count': len(actual_hist),
                'average': sum(actual_hist) / len(actual_hist) if actual_hist else 0,
                'max': max(actual_hist) if actual_hist else 0,
                'min': min(actual_hist) if actual_hist else 0
            },
            'future_stats': {
                'count': len(future_preds),
                'average': sum(future_preds) / len(future_preds) if future_preds else 0,
                'max': max(future_preds) if future_preds else 0,
                'min': min(future_preds) if future_preds else 0
            }
        }
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze this banking branch performance data and provide comprehensive insights:

        Dataset Information: {dataset_info}
        
        Historical Data: {data_summary['historical_data']}
        Historical Statistics: {data_summary['historical_stats']}
        
        Future Predictions: {data_summary['future_predictions']}
        Future Statistics: {data_summary['future_stats']}
        
        Please provide:
        1) A summary of the data and predictions
        2) A detailed analysis of the data and predictions
        3) A list of key insights and recommendations
        4) A list of suggested follow-up questions

        Note: All currency amounts are displayed in NPR (Nepalese Rupees).
        
        Format the response with clear sections and bullet points. Focus on actionable insights for banking operations.
        
        At the end, include a section called "Suggested Follow-up Questions" with 3-5 specific questions that users could ask to dive deeper into the analysis, such as:
        - "What factors are driving the predicted growth trend?"
        - "How does this branch compare to other branches in similar regions?"
        - "What seasonal patterns should we expect in the coming months?"
        - "What strategies could improve performance based on these predictions?"
        """
        
        # Generate response using Gemini
        response = model_gemini.generate_content(prompt)
        
        # Format the response
        insights = []
        insights.append("🤖 **AI-Generated Insights (Powered by Gemini)**")
        insights.append("=" * 50)
        insights.append(response.text)
        insights.append("\n" + "=" * 50)
        insights.append("💡 **Tip**: Ask follow-up questions to explore specific aspects of the analysis!")
        
        return "\n".join(insights)
        
    except ImportError:
        return "Error: Gemini API not available. Please install google-generativeai package."
    except Exception as e:
        return f"Error generating insights with Gemini: {str(e)}"

def handle_followup_question(question, months_hist, actual_hist, future_months, future_preds, dataframes):
    """
    Handle follow-up questions based on the prediction results and available data
    
    Args:
        question: User's follow-up question
        months_hist: Historical months
        actual_hist: Historical actual values
        future_months: Future months for prediction
        future_preds: Predicted values for future months
        dataframes: Available dataframes for additional context
    
    Returns:
        str: AI-generated response to the follow-up question
    """
    try:
        dataset_info = get_dataset_info()

        # Prepare comprehensive context for the follow-up question
        context = {
            'historical_data': list(zip(months_hist, actual_hist)),
            'future_predictions': list(zip(future_months, future_preds)),
            'historical_stats': {
                'count': len(actual_hist),
                'average': sum(actual_hist) / len(actual_hist) if actual_hist else 0,
                'max': max(actual_hist) if actual_hist else 0,
                'min': min(actual_hist) if actual_hist else 0
            },
            'future_stats': {
                'count': len(future_preds),
                'average': sum(future_preds) / len(future_preds) if future_preds else 0,
                'max': max(future_preds) if future_preds else 0,
                'min': min(future_preds) if future_preds else 0
            },
            'available_datasets': list(dataframes.keys())
        }
        
        # Create prompt for follow-up question
        prompt = f"""
        Based on the following banking data and prediction results, please answer this follow-up question:
        
        User Question: {question}
        
        Available Context:
        - Dataset Information: {dataset_info}
        - Historical Data: {context['historical_data']}
        - Historical Statistics: {context['historical_stats']}
        - Future Predictions: {context['future_predictions']}
        - Future Statistics: {context['future_stats']}
        - Available Datasets: {context['available_datasets']}
        
        Please provide a detailed response that:
        1. Directly addresses the user's question
        2. Uses the available data and predictions
        3. Provides actionable insights
        4. Suggests additional analysis if relevant

        Note: All currency amounts are displayed in NPR (Nepalese Rupees).
        Note: Don't write techincal responses like code or suggestions to analyze the data, you analyze yourself and provice business insights.
        
        Format your response clearly with bullet points and sections as appropriate.
        """
        
        # Generate response using Gemini
        response = model_gemini.generate_content(prompt)
        
        # Format the response
        followup_response = []
        followup_response.append("🤖 **Follow-up Analysis**")
        followup_response.append("=" * 40)
        followup_response.append(f"**Question**: {question}")
        followup_response.append("=" * 40)
        followup_response.append(response.text)
        
        return "\n".join(followup_response)
        
    except Exception as e:
        return f"Error processing follow-up question: {str(e)}"

def interactive_insights_session(months_hist, actual_hist, future_months, future_preds):
    """
    Interactive session for generating insights and handling follow-up questions
    
    Args:
        months_hist: Historical months
        actual_hist: Historical actual values
        future_months: Future months for prediction
        future_preds: Predicted values for future months
    """
    print("🔍 Generating initial insights...")
    initial_insights = generate_insights(months_hist, actual_hist, future_months, future_preds)
    print(initial_insights)
    
    print("\n" + "=" * 60)
    print("💬 **Interactive Follow-up Session**")
    print("Ask follow-up questions to dive deeper into the analysis!")
    print("Type 'done' to exit the session")
    print("=" * 60)
    
    while True:
        try:
            followup_question = input("\n❓ Your follow-up question: ").strip()
            
            if followup_question.lower() in ['done', 'exit', 'quit', '']:
                print("👋 Ending insights session. Thank you!")
                break
            
            print("\n🤖 Processing your follow-up question...")
            followup_response = handle_followup_question(
                followup_question, 
                months_hist, 
                actual_hist, 
                future_months, 
                future_preds, 
                dataframes
            )
            print(f"\n{followup_response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Ending insights session. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

# Interactive_insights_session(
#     months_hist=['2023-01', '2023-02', '2023-03'],
#     actual_hist=[100000, 120000, 110000],
#     future_months=['2023-04', '2023-05', '2023-06'],
#     future_preds=[130000, 140000, 150000]
# )
# Uncomment the above line to start an interactive insights session with sample data
# Note: Replace the sample data with actual historical and predicted values from predict.py.  