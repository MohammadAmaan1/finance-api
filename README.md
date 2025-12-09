# Loan Approval Prediction API (FastAPI + Render Deployment)

## 1. Project Overview
This project is a Machine Learning-based Loan Approval Prediction system.  
The ML model is served using FastAPI and deployed on Render.com, making it accessible via a live URL.

Main features:
- Predicts loan approval (Approved / Rejected)
- Returns risk probability (0–1)
- Includes a server health check endpoint
- Provides model metadata (features list, feature count)

---

## 2. Tech Stack
- Python  
- scikit-learn  
- FastAPI  
- Uvicorn  
- Pandas / NumPy  
- Render (Deployment)

---

## 3. Folder Structure

Finance_API/
│── main.py
│── final_model.pkl
│── requirements.txt
│── README.md


---

## 4. How To Run Locally

### Step 1 — Clone or download the folder
git clone <REPO_URL>
cd Finance_API


### Step 2 — Create virtual environment (optional)
Windows:
python -m venv venv
venv\Scripts\activate


### Step 3 — Install dependencies
pip install -r requirements.txt


### Step 4 — Run FastAPI server
uvicorn main:app --reload


### Step 5 — Open in browser
- Swagger UI:  http://127.0.0.1:8000/docs  
- Home:        http://127.0.0.1:8000/

---

## 5. Live API (Render Deployment)

**Base URL:**  
`https://YOUR-RENDER-URL.onrender.com`

Replace `YOUR-RENDER-URL` with your actual deployed API URL.

Endpoints available:
- `/health` — Check server status  
- `/info` — Get model metadata  
- `/predict` — Make loan predictions  

---

## 6. API Endpoints Detail

### ✔ `/health` (GET)
Checks if the server is running.
Example response:
```json
{ "status": "healthy" }
