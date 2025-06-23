# -*- coding: utf-8 -*-
import pandas as pd
import google.generativeai as genai
import joblib
import os
import json
import numpy as np

# --- CONFIGURATION ---
CSV_FILES = {
    'branch': 'data/global_ime_enhanced_branch_data.csv',
    'performance': 'data/global_ime_monthly_performance_enhanced.csv',
    'product_performance': 'data/global_ime_product_performance.csv',
    'customer_segments': 'data/global_ime_customer_segments.csv',
    'monthly_product': 'data/global_ime_monthly_product_data.csv',
}

# Note: All currency data is already in NPR, just need to display correctly

MODEL_PATH = 'modelsEachBranch/branch_4_features.pkl'
GEMINI_API_KEY = 'AIzaSyANUOrGo-odqdIvJHJCVvlDzGIWn95cIe4'

# --- LOAD DATA ---
dataframes = {}
for name, path in CSV_FILES.items():
    if os.path.exists(path):
        dataframes[name] = pd.read_csv(path)
        print(f"✓ Loaded {name}: {path} ({len(dataframes[name])} rows)")
    else:
        print(f"✗ Warning: {path} not found. Skipping.")

print(f"\nLoaded {len(dataframes)} CSV files successfully!")

# Print column information for debugging
for name, df in dataframes.items():
    print(f"\n{name} columns: {list(df.columns)}")

# --- LOAD MODEL ---
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model file {MODEL_PATH} not found. Prediction feature will be disabled.")

# --- SETUP GEMINI ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
else:
    model_gemini = None
    print("Warning: GEMINI_API_KEY not set. LLM features will be disabled.")

def format_amount_npr(amount):
    """Format amount that is already in NPR"""
    if pd.isna(amount) or amount == 0:
        return 0
    return amount  # Amount is already in NPR, no conversion needed

def format_currency_npr(amount):
    """Format amount in NPR currency"""
    if pd.isna(amount) or amount == 0:
        return "NPR 0"
    return f"NPR {amount:,.2f}"

def get_dataset_info(dataframes):
    """Get detailed information about available datasets and their columns"""
    dataset_info = {}
    for name, df in dataframes.items():
        dataset_info[name] = {
            'columns': list(df.columns),
            'row_count': len(df),
            'description': get_dataset_description(name),
            'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
        }
    return dataset_info

def get_dataset_description(dataset_name):
    """Get description of what each dataset contains"""
    descriptions = {
        'branch': 'Branch information including location, staff count, revenue, profit, customer counts, digital adoption rates, and financial metrics',
        'performance': 'Monthly performance data with revenue, expenses, profit, transactions, and digital metrics by branch',
        'product_performance': 'Product-specific performance data including customer base, growth rates, satisfaction, and revenue per customer',
        'customer_segments': 'Customer demographic and segmentation data with deposit, loan, and digital adoption information by branch',
        'monthly_product': 'Monthly product usage and performance data including transaction counts, volumes, and revenue by product category'
    }
    return descriptions.get(dataset_name, 'Dataset description not available')

def analyze_question_with_llm(question, dataframes):
    """Step 1: Use LLM to analyze the question and understand what needs to be done"""
    if not model_gemini:
        return create_fallback_analysis(question, list(dataframes.keys()))
    
    dataset_info = get_dataset_info(dataframes)
    dataset_details = ""
    for name, info in dataset_info.items():
        dataset_details += f"\n{name}:\n"
        dataset_details += f"  Description: {info['description']}\n"
        dataset_details += f"  Columns: {', '.join(info['columns'][:10])}{'...' if len(info['columns']) > 10 else ''}\n"
        dataset_details += f"  Rows: {info['row_count']}\n"
    
    analysis_prompt = f"""
You are a banking data analyst. Analyze this question and determine what type of analysis is needed.

Question: "{question}"

Available datasets and their details:
{dataset_details}

Important notes:
- For revenue/financial questions, check 'branch' dataset (has total_revenue, profit columns) and 'performance' dataset (has monthly revenue)
- For branch-wise analysis, use 'branch' dataset or 'performance' dataset
- For customer questions, use 'customer_segments' or 'branch' dataset (has customer counts)
- For product questions, use 'product_performance' or 'monthly_product' datasets
        - All currency amounts are already in NPR (Nepalese Rupees), just need proper display formatting

Respond with ONLY a valid JSON object in this exact format:

{{
    "question_type": "company_wide_metrics",
    "required_datasets": ["branch", "performance"],
    "required_operations": ["group_by_branch", "sum_revenue", "format_currency"],
    "grouping_columns": ["branch_name"],
    "aggregation_columns": ["total_revenue", "profit"],
    "filters": {{}},
    "business_context": "User wants to know total revenue from each branch",
    "analysis_approach": "Group by branch_name and sum total_revenue, then convert to NPR"
}}

Question types:
- company_wide_metrics (totals, averages across all branches)
- branch_specific (data for a particular branch)
- branch_comparison (comparing multiple branches)
- product_analysis (product performance, categories)
- customer_analysis (customer segments, demographics)
- trend_analysis (changes over time, growth patterns)
- financial_analysis (revenue, profit, expenses)

Required operations can include:
- group_by_branch, group_by_product, group_by_segment
- sum_revenue, sum_profit, sum_customers, count_branches
- format_currency, calculate_percentage, calculate_average
- filter_by_date, filter_by_branch, filter_by_product

Only respond with the JSON object, nothing else.
"""
    
    try:
        response = model_gemini.generate_content(analysis_prompt)
        response_text = response.text.strip()
        
        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            print(f"Warning: LLM returned invalid JSON: {response_text[:200]}...")
            return create_fallback_analysis(question, list(dataframes.keys()))
            
    except Exception as e:
        print(f"Error in LLM analysis: {str(e)}")
        return create_fallback_analysis(question, list(dataframes.keys()))

def create_fallback_analysis(question, available_datasets):
    """Create a fallback analysis when LLM fails"""
    question_lower = question.lower()
    
    # Revenue analysis
    if any(word in question_lower for word in ['revenue', 'income', 'earning']) and any(word in question_lower for word in ['branch', 'each']):
        return {
            "question_type": "financial_analysis",
            "required_datasets": ["branch"],
            "required_operations": ["group_by_branch", "sum_revenue", "format_currency"],
            "grouping_columns": ["branch_name"],
            "aggregation_columns": ["total_revenue"],
            "filters": {},
            "business_context": "User wants revenue by branch",
            "analysis_approach": "Group by branch and sum revenue"
        }
    
    # Customer analysis
    if any(word in question_lower for word in ['customer', 'customers']):
        return {
            "question_type": "customer_analysis",
            "required_datasets": ["branch", "customer_segments"],
            "required_operations": ["sum_customers", "group_by_branch"],
            "grouping_columns": ["branch_name"],
            "aggregation_columns": ["total_customers", "customer_count"],
            "filters": {},
            "business_context": "User wants customer information",
            "analysis_approach": "Analyze customer data"
        }
    
    # Branch analysis
    if any(word in question_lower for word in ['branch', 'branches']):
        return {
            "question_type": "company_wide_metrics",
            "required_datasets": ["branch"],
            "required_operations": ["count_branches", "sum_revenue"],
            "grouping_columns": [],
            "aggregation_columns": ["total_revenue"],
            "filters": {},
            "business_context": "User wants branch information",
            "analysis_approach": "Analyze branch data"
        }
    
    # Default analysis
    return {
        "question_type": "comprehensive_analysis",
        "required_datasets": available_datasets[:2],
        "required_operations": ["basic_summary"],
        "grouping_columns": [],
        "aggregation_columns": [],
        "filters": {},
        "business_context": "General inquiry",
        "analysis_approach": "Provide general overview"
    }

def execute_data_operations(analysis_result, dataframes):
    """Execute the required data operations based on analysis"""
    try:
        required_datasets = analysis_result.get('required_datasets', [])
        required_operations = analysis_result.get('required_operations', [])
        grouping_columns = analysis_result.get('grouping_columns', [])
        aggregation_columns = analysis_result.get('aggregation_columns', [])
        filters = analysis_result.get('filters', {})
        
        results = {}
        
        # Process each required dataset
        for dataset_name in required_datasets:
            if dataset_name not in dataframes:
                continue
                
            df = dataframes[dataset_name].copy()
            
            # Apply filters
            for column, value in filters.items():
                if column in df.columns:
                    df = df[df[column] == value]
            
            # Execute specific operations
            if 'group_by_branch' in required_operations:
                results.update(execute_branch_grouping(df, aggregation_columns))
            
            if 'sum_revenue' in required_operations:
                results.update(execute_revenue_operations(df))
            
            if 'sum_customers' in required_operations:
                results.update(execute_customer_operations(df))
            
            if 'count_branches' in required_operations:
                results.update(execute_branch_counting(df))
            
            if 'product_analysis' in required_operations:
                results.update(execute_product_analysis(df))
            
            if 'trend_analysis' in required_operations:
                results.update(execute_trend_analysis(df))
            
            if 'basic_summary' in required_operations:
                results.update(execute_basic_summary(df, dataset_name))
        
        # Format currency if needed
        if 'format_currency' in required_operations:
            results = format_currency_in_results(results)
        
        return results
        
    except Exception as e:
        print(f"Error in data operations: {str(e)}")
        return {"error": f"Failed to execute data operations: {str(e)}"}

def execute_branch_grouping(df, aggregation_columns):
    """Group data by branch and aggregate specified columns"""
    results = {}
    
    # Find the branch name column
    branch_col = None
    for col in df.columns:
        if 'branch' in col.lower() and 'name' in col.lower():
            branch_col = col
            break
    
    if not branch_col:
        # Try other possible branch identifier columns
        for col in ['branch_name', 'branch', 'name']:
            if col in df.columns:
                branch_col = col
                break
    
    if branch_col and branch_col in df.columns:
        # Group by branch and aggregate
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Focus on revenue-related columns
        revenue_columns = [col for col in numeric_columns if any(keyword in col.lower() 
                          for keyword in ['revenue', 'income', 'profit', 'earning'])]
        
        if revenue_columns:
            branch_grouped = df.groupby(branch_col)[revenue_columns].sum().round(2)
            results['branch_revenue'] = branch_grouped.to_dict('index')
        
        # Also include customer-related columns
        customer_columns = [col for col in numeric_columns if any(keyword in col.lower() 
                           for keyword in ['customer', 'client'])]
        
        if customer_columns:
            branch_customers = df.groupby(branch_col)[customer_columns].sum()
            results['branch_customers'] = branch_customers.to_dict('index')
    
    return results

def execute_revenue_operations(df):
    """Execute revenue-related operations"""
    results = {}
    
    # Find revenue columns
    revenue_columns = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['revenue', 'income', 'earning'])]
    
    if revenue_columns:
        for col in revenue_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                total_revenue = df[col].sum()
                results[f'total_{col.lower()}'] = total_revenue
                
                # Also provide branch-wise breakdown if branch column exists
                branch_col = find_branch_column(df)
                if branch_col:
                    branch_revenue = df.groupby(branch_col)[col].sum().to_dict()
                    results[f'{col.lower()}_by_branch'] = branch_revenue
    
    return results

def execute_customer_operations(df):
    """Execute customer-related operations"""
    results = {}
    
    # Find customer columns
    customer_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['customer', 'client', 'account'])]
    
    if customer_columns:
        for col in customer_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                total_customers = df[col].sum()
                results[f'total_{col.lower()}'] = total_customers
                
                # Branch-wise breakdown
                branch_col = find_branch_column(df)
                if branch_col:
                    branch_customers = df.groupby(branch_col)[col].sum().to_dict()
                    results[f'{col.lower()}_by_branch'] = branch_customers
    
    return results

def execute_branch_counting(df):
    """Count branches and provide branch statistics"""
    results = {}
    if 'branch_id' in df.columns:
        results['total_branches'] = df['branch_id'].nunique()
        results['branch_ids'] = df['branch_id'].unique().tolist()
    else:
        branch_col = find_branch_column(df)
        if branch_col:
            unique_branches = df[branch_col].nunique()
            results['total_branches'] = unique_branches
            results['branch_names'] = df[branch_col].unique().tolist()
    return results

def execute_product_analysis(df):
    """Execute product-related analysis"""
    results = {}
    
    # Find product columns
    product_columns = [col for col in df.columns if 'product' in col.lower()]
    
    if 'product_name' in df.columns:
        results['total_products'] = df['product_name'].nunique()
        results['product_list'] = df['product_name'].unique().tolist()
    
    if 'category' in df.columns:
        results['product_categories'] = df['category'].value_counts().to_dict()
    
    return results

def execute_trend_analysis(df):
    """Execute trend analysis if time-based data exists"""
    results = {}
    
    # Look for date/month columns
    time_columns = [col for col in df.columns if any(keyword in col.lower() 
                   for keyword in ['date', 'month', 'year', 'time'])]
    
    if time_columns and len(time_columns) > 0:
        time_col = time_columns[0]
        
        # Find numeric columns for trend analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for num_col in numeric_columns:
            if 'revenue' in num_col.lower() or 'profit' in num_col.lower():
                trend_data = df.groupby(time_col)[num_col].sum().to_dict()
                results[f'{num_col.lower()}_trend'] = trend_data
    
    return results

def execute_basic_summary(df, dataset_name):
    """Provide basic summary of the dataset"""
    results = {}
    
    results[f'{dataset_name}_summary'] = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns)
    }
    
    # Get some key statistics
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        results[f'{dataset_name}_key_stats'] = {
            'sum': numeric_df.sum().to_dict(),
            'mean': numeric_df.mean().to_dict(),
            'count': numeric_df.count().to_dict()
        }
    
    return results

def find_branch_column(df):
    """Find the column that contains branch information"""
    possible_columns = ['branch_name', 'branch', 'name', 'branch_id']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # Look for columns containing 'branch'
    for col in df.columns:
        if 'branch' in col.lower():
            return col
    
    return None

def format_currency_in_results(results):
    """Format currency amounts that are already in NPR"""
    formatted_results = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (like branch-wise data)
            formatted_dict = {}
            for sub_key, sub_value in value.items():
                # Keep the original NPR amounts, just ensure proper formatting
                formatted_dict[sub_key] = sub_value
            formatted_results[key] = formatted_dict
        else:
            # Keep original amounts (already in NPR)
            formatted_results[key] = value
    
    return formatted_results

def generate_insights_with_llm(question, analysis_result, calculation_results, dataframes):
    """Step 4: Use LLM to generate insights and final response"""
    if not model_gemini:
        return generate_fallback_response(question, analysis_result, calculation_results)
    
    results_summary = json.dumps(calculation_results, indent=2, default=str)[:2000]  # Limit size
    
    insight_prompt = f"""
You are a banking business analyst. Provide a clear, business-friendly answer to this question.

Question: {question}

Analysis Type: {analysis_result.get('question_type', 'Unknown')}
Business Context: {analysis_result.get('business_context', 'Not specified')}

Calculation Results: {results_summary}

Important formatting instructions:
- All currency amounts are already in NPR (Nepalese Rupees), no conversion needed
- Format currency as "NPR X,XXX.XX" for display
- Use proper number formatting with commas for thousands
- Provide specific numbers from the data
- If showing branch-wise data, format it clearly
- Be direct and specific in your answer
- Focus on the exact question asked

Respond with a clear, professional answer without JSON formatting.
"""
    
    try:
        response = model_gemini.generate_content(insight_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return generate_fallback_response(question, analysis_result, calculation_results)

def generate_fallback_response(question, analysis_result, calculation_results):
    """Generate a fallback response when LLM fails"""
    if not calculation_results:
        return "I couldn't find specific data to answer your question. Please try asking about revenue, customers, branches, or products."
    
    response_parts = []
    question_lower = question.lower()
    
    # Handle revenue questions
    if any(keyword in question_lower for keyword in ['revenue', 'income', 'earning']):
        if 'branch_revenue' in calculation_results:
            response_parts.append("Revenue by branch:")
            branch_data = calculation_results['branch_revenue']
            for branch, data in branch_data.items():
                if isinstance(data, dict):
                    for metric, value in data.items():
                        if 'revenue' in metric.lower():
                            response_parts.append(f"• {branch}: {format_currency_npr(value)}")
                else:
                    response_parts.append(f"• {branch}: {format_currency_npr(data)}")
        
        # Also check for total revenue
        for key, value in calculation_results.items():
            if 'total' in key.lower() and 'revenue' in key.lower() and isinstance(value, (int, float)):
                response_parts.append(f"Total Revenue: {format_currency_npr(value)}")
    
    # Handle customer questions
    if any(keyword in question_lower for keyword in ['customer', 'client']):
        if 'branch_customers' in calculation_results:
            response_parts.append("Customers by branch:")
            branch_data = calculation_results['branch_customers']
            for branch, data in branch_data.items():
                if isinstance(data, dict):
                    total_customers = sum(v for v in data.values() if isinstance(v, (int, float)))
                    response_parts.append(f"• {branch}: {total_customers:,} customers")
                else:
                    response_parts.append(f"• {branch}: {data:,} customers")
    
    # Handle branch questions
    if 'total_branches' in calculation_results:
        response_parts.append(f"Total Branches: {calculation_results['total_branches']}")
    
    if 'branch_names' in calculation_results:
        branches = calculation_results['branch_names']
        response_parts.append(f"Branch Names: {', '.join(branches[:10])}")
        if len(branches) > 10:
            response_parts.append(f"... and {len(branches) - 10} more branches")
    
    # If no specific patterns matched, show available data
    if not response_parts:
        response_parts.append("Analysis Results:")
        for key, value in calculation_results.items():
            if isinstance(value, (int, float)):
                if any(keyword in key.lower() for keyword in ['revenue', 'income', 'profit']):
                    response_parts.append(f"• {key.replace('_', ' ').title()}: {format_currency_npr(value)}")
                else:
                    response_parts.append(f"• {key.replace('_', ' ').title()}: {value:,}")
            elif isinstance(value, dict) and len(value) <= 5:
                response_parts.append(f"• {key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(response_parts) if response_parts else "No specific data found for your query."

def intelligent_chatbot_response(question, dataframes):
    """Main function that orchestrates the intelligent analysis process"""
    try:
        # Step 1: LLM-based question analysis
        print("🔍 Analyzing question with LLM...")
        analysis_result = analyze_question_with_llm(question, dataframes)
        
        if isinstance(analysis_result, str) and "Error" in analysis_result:
            return analysis_result
        
        print(f"📊 Analysis: {analysis_result.get('question_type', 'Unknown')} - {analysis_result.get('business_context', 'No context')}")
        
        # Step 2: Execute data operations
        print("🧮 Executing data operations...")
        calculation_results = execute_data_operations(analysis_result, dataframes)
        
        if 'error' in calculation_results:
            return calculation_results['error']
        
        print(f"💾 Found {len(calculation_results)} result categories")
        
        # Step 3: Generate insights with LLM
        print("💡 Generating insights...")
        final_response = generate_insights_with_llm(question, analysis_result, calculation_results, dataframes)
        
        return final_response
        
    except Exception as e:
        return f"Error in intelligent analysis: {str(e)}"

def main():
    """Main chat loop"""
    print("🏦 Welcome to the Improved Banking Data Chatbot!")
    print("=" * 60)
    print("This chatbot uses LLM to understand your questions and provide intelligent analysis.")
    print("All currency amounts are displayed in NPR (data is already in NPR, not USD).")
    print("\nAvailable data:")
    for name, df in dataframes.items():
        print(f"• {name}: {len(df)} records")
    print("\nExample questions:")
    print("• 'Give me total revenue from each branch'")
    print("• 'How many customers do we have in total?'")
    print("• 'What is the profit by branch?'")
    print("• 'Show me branch performance data'")
    print("\nType 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n💬 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using the Banking Data Chatbot!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Bot: Analyzing your question...")
            response = intelligent_chatbot_response(question, dataframes)
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()