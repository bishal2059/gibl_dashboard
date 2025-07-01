from fastapi.middleware.cors import CORSMiddleware
from predict import get_prediction_of_branch_n
from fastapi import FastAPI, HTTPException
from chatbot import data_analysis_chat
from chatbot import generate_insights
from parseToJson import extract_insights_json
from datetime import datetime
from pydantic import BaseModel
from typing import List
from chatbot import generate_followup_resposne
import json
from pathlib import Path
import random
import csv

app = FastAPI()

# Allow frontend origin here
origins = [
    "http://localhost:5173", 
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # allow specific origins
    allow_credentials=True,
    allow_methods=["*"],               # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],               # allow all headers
)


DATA_FILE = Path("data_store.json")


def save_dict(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


def load_dict() -> dict:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

@app.get("/")
def read_root():
    return {"Hello": "World"}

def format_month(month_str: str) -> str:
    """Convert 'YYYY-MM' → 'Mon YYYY' (e.g., '2024-01' → 'Jan 2024')"""
    try:
        dt = datetime.strptime(month_str, "%Y-%m")
        return dt.strftime("%b %Y")
    except ValueError:
        return month_str  # fallback if format is already okay

def calculate_growth(revenues: List[float], predict=False, last_historical = None) -> List[float]:
    """Calculate percent growth month over month."""
    if predict:
        growths = []
        # For predicted revenue, we calculate first month growth using last historical month
        for i in range(len(revenues)):
            prev = revenues[i - 1] if i > 0 else last_historical
            current = revenues[i]
            if prev == 0:
                growth = 0.0
            else:
                growth = round(((current - prev) / prev) * 100, 1)
            growths.append(growth)
    else:
        growths = [0.0]  # first month has no growth
        for i in range(1, len(revenues)):
            prev = revenues[i - 1]
            current = revenues[i]
            if prev == 0:
                growth = 0.0
            else:
                growth = round(((current - prev) / prev) * 100, 1)
            growths.append(growth)
    
    return growths

def deduplicate_months(data):
    seen = set()
    unique = []
    for item in data:
        if item["month"] not in seen:
            unique.append(item)
            seen.add(item["month"])
    return unique

@app.get("/api/predict/{branch_Id}/{months}")
def get_branch_forecast(branch_Id: str, months: int = 6):
    try:
        data = {}
        with open("unique_branch.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["branch_id"] == branch_Id[7:]:
                    data = {
                        "id": branch_Id,
                        "name": row["branch_name"],
                        "location": row["district_x"],
                        "province": row["province_x"]
                    }
                    break
            months_hist, actual_hist, future_months, future_preds = get_prediction_of_branch_n(data["name"], months)

            #save data into file
            user_id = f"user_{branch_Id}"
            server_data = {
                "months_hist": months_hist,
                "actual_hist": actual_hist,
                "future_months": future_months,
                "future_preds": future_preds
            }
            save_dict({user_id: server_data})
            

            # Format historical revenue
            historical_growths = calculate_growth(actual_hist)
            historicalRevenue_raw = [
                {
                    "month": format_month(month),
                    "revenue": round(revenue),
                    "growth": growth
                }
                for month, revenue, growth in zip(months_hist, actual_hist, historical_growths)
            ]

            historicalRevenue = deduplicate_months(historicalRevenue_raw)

            # Format predicted revenue
            predicted_growths = calculate_growth(future_preds, predict=True, last_historical=historicalRevenue_raw[-1]["revenue"] if historicalRevenue_raw else 0)
            predictedRevenue = [
                {
                    "month": format_month(month),
                    "revenue": round(revenue),
                    "growth": growth,
                    "confidence": round(random.uniform(80, 95), 2)
                }
                for month, revenue, growth in zip(future_months, future_preds, predicted_growths)
            ]
            data["historicalRevenue"] = historicalRevenue
            data["predictedRevenue"] = predictedRevenue
            return data
            
        raise HTTPException(status_code=404, detail="Branch not found")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="CSV file not found")

class ChatQuestion(BaseModel):
    question: str

@app.post("/api/chatbot/insights")
def chat_response(qns: ChatQuestion):
    """Handle chat questions for data analysis."""
    # print(qns.question)
    try:
        answer = data_analysis_chat(qns.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/current_insights/{branch_Id}")
def get_current_insights(branch_Id: str):
    """Generate and return current insights."""
    try:
        #get data from redis
        print(f"Fetching insights for branch: {branch_Id}")
        user_id = f"user_{branch_Id}"
        data = load_dict().get(user_id, {})
        if not data:
            raise HTTPException(status_code=404, detail="No data found for this branch")
        
        server_data = data

        insights = generate_insights(server_data.get('months_hist', []),
                                     server_data.get('actual_hist', []),
                                     server_data.get('future_months', []),
                                     server_data.get('future_preds', []))
        # print(insights)
        return extract_insights_json(insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

class ChatInsightQuestion(BaseModel):
    question: str
    branchId: str

@app.post("/api/chat_insights")
def chat_insights(qns: ChatInsightQuestion):
    """Handle chat questions for insights."""
    try:
        # Extract branch ID from the question
        branch_id = qns.branchId
        if not branch_id:
            raise HTTPException(status_code=400, detail="Branch ID is required")
        # Get the data for the specific branch
        user_id = f"user_{branch_id}"
        data = load_dict().get(user_id, {})
        if not data:
            raise HTTPException(status_code=404, detail="No data found for this branch")
        # Call the chat function with the question and data
        answer = generate_followup_resposne(qns.question, data.get('months_hist', []),
                                            data.get('actual_hist', []),
                                            data.get('future_months', []),
                                            data.get('future_preds', []))
        if not answer:
            raise HTTPException(status_code=404, detail="No insights found for this question")
        # Return the answer
        return {"message": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/api/kpi")
def get_kpi_values():
    # retrieve KPI values from the database or any other source
    data = {
            "total_revenue": "NPR 45.2M",
            "total_revenue_change": "+8.3%",
            "active_branches": "125",
            "active_branches_change": "+5",
            "total_customers": "1.2M",
            "total_customers_change": "+12.5%",
            "growth_rate": "12.8%",
            "growth_rate_change": "-2.1%"
            }
    try:
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching KPI values: {str(e)}")
    
