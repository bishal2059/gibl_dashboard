# To run locally:

## Add the following folders and files:
1. **data** folder at backend/
2. **modelsEachBranch** at backend/
3. **unique_branch.csv** at backend/
4. **.env** at backend/ with necessary API key as per .env.example

## Run server and client

1. Start server:
```
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

2. Run client:
```
cd frontend
npm install
npm run dev
```