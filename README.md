## To run locally:

1. Start server:
```
cd backend
python -m venv .venv
pip install -r requirements.txt
uvicorn main:app --reload
```

2. Run client:
```
cd gibl_ui
npm install
npm run dev
```