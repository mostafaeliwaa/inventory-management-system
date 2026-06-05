<div align="center">

# 🚀 AI-Powered Retail Inventory Management System

**A next-generation, data-driven solution to revolutionize retail distribution across 13 branches in Egypt.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Machine_Learning-orange?style=for-the-badge)](#)
[![Prophet](https://img.shields.io/badge/Prophet-Time_Series-blue?style=for-the-badge)](#)

*Transforming data into actionable insights, predicting revenue, and optimizing stock distribution with cutting-edge AI.*

</div>

---

## 🌟 Overview

The **Inventory Management System** is an end-to-end robust platform engineered to solve real-world retail hurdles. By fusing **Advanced Data Analytics**, **Machine Learning**, **Time-Series Forecasting**, and **Generative AI**, this system transcends traditional inventory tracking. It provides predictive insights, optimizes stock levels, and empowers decision-makers to maximize profitability.

### 🎯 Key Objectives
- **📉 Minimize Dead Stock:** Proactively identify non-moving inventory before it erodes profit margins.
- **🚚 Optimize Distribution:** Classify product demand (Fast/Medium/Slow) to align inventory with branch-specific needs.
- **💸 Forecast Revenue:** Predict daily and weekly branch revenue accurately for smarter staffing, cash flow, and logistics planning.
- **📄 Automate Invoicing:** Instantly digitize and process paper invoices using AI-driven OCR.

---

## 🔥 Key Features

### 📊 1. Advanced Analytics & Dashboards
- **ABC Classification:** Intelligently segment 2,400+ products by sales volume and revenue.
- **Branch Performance Tracking:** Compare and analyze KPIs across 13 distinct retail branches.
- **Anomaly Detection:** Automatically flag unusual stock-to-sales ratios for rapid intervention.
- **Interactive Visualizations:** Explore responsive, insight-rich dashboards powered by Plotly.

### 🧠 2. Machine Learning Suite (`/ML models/`)
- **Demand Classification:** A powerful Random Forest model predicting Fast/Medium/Slow moving items *(Macro F1: ~0.90)*.
- **Dead-Stock Predictor:** A Gradient Boosting model pinpointing static inventory with extreme precision *(ROC-AUC: 0.998)*.
- **Interactive UI:** Built-in Streamlit app (`port 8501`) for effortless manual testing and batch inference.

### 📈 3. Time-Series Forecasting (`/forecasting/`)
Built on 144 days of POS data across 8 branches, adeptly handling Ramadan/Eid shocks, weekly cycles, and payday windows.
- **Daily Forecasts (7-day horizon):** Prophet + LightGBM Ensemble *(WAPE: 0.479)*.
- **Weekly Forecasts (4-week horizon):** Ensemble (Prophet + LGBM + Naive Mean) for robust cash-flow planning *(WAPE: 0.229)*.

### 🤖 4. AI Assistant & Smart OCR
- **Smart OCR Pipeline (`/ocr1/`):** Digitize physical invoices and seamlessly match products against the Supabase database.
- **AI Chatbot (`/chatbot/`):** LangGraph-powered virtual assistant ready to answer complex inventory queries in natural language.
- **Unified API Gateway:** Everything is served elegantly through a single FastAPI application (`main.py`).

---

## 🏗️ Architecture & Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Core & Data Prep** | `Python`, `Pandas`, `NumPy`, `Scikit-learn` |
| **Machine Learning** | `LightGBM`, `Random Forest`, `Gradient Boosting` |
| **Time Series** | `Prophet`, `LightGBM (Recursive & Direct)` |
| **APIs & Backend** | `FastAPI`, `Uvicorn`, `Pydantic`, `Supabase` |
| **Frontend & BI** | `Streamlit`, `Plotly`, `Jupyter Notebooks` |
| **AI / NLP** | `LangGraph`, `Google Gemini API` |

---

## 🏁 Quick Start

### Prerequisites
- Python 3.8+ installed.
- Supabase credentials and Gemini API Key configured in `.env`.

### 1. Clone & Install
```bash
git clone https://github.com/mostafaeliwaa/inventory-management-system.git
cd inventory-management-system
pip install -r requirements.txt
```

### 2. Run the Unified Backend Server
Launch the central FastAPI server that powers the OCR, Chatbot, and Forecasting modules.
```bash
uvicorn main:app --port 8000 --reload
# Access the interactive API docs at http://localhost:8000/docs
```

### 3. Launch the ML Streamlit Interface
```bash
cd "ML models"
pip install -r requirements.txt
streamlit run streamlit_app.py
# The UI will open at http://localhost:8501
```

### 4. Explore the Data Analytics
```bash
cd analysis
jupyter notebook EXP.ipynb
```

---

## 📁 Repository Structure

```text
📦 inventory-management-system
 ┣ 📂 analysis/        # Jupyter notebooks & Plotly HTML dashboards
 ┣ 📂 cleaning/        # ETL pipelines and data sanitization scripts
 ┣ 📂 data/            # Cleaned CSV files and database schemas
 ┣ 📂 ML models/       # Classification & Dead-Stock models (FastAPI + Streamlit)
 ┣ 📂 forecasting/     # Time-Series revenue forecasting (Prophet + LGBM)
 ┣ 📂 chatbot/         # LangGraph-based AI Chatbot
 ┣ 📂 ocr1/            # OCR extraction and database matching scripts
 ┣ 📂 routers/         # FastAPI routers for sub-services
 ┣ 📜 main.py          # Main Unified API application entry point
 ┗ 📜 requirements.txt # Global dependencies
```

---

## 🤝 Contributing & Documentation

Detailed documentation for each module can be found in their respective directories:
- 📖 [Project Guidelines (CLAUDE.md)](CLAUDE.md)
- 📊 [Analytics Documentation](analysis/README.md)
- 🧠 [Machine Learning Documentation](ML%20models/README.md)
- 📈 [Forecasting Documentation](forecasting/README.md)
- 🧹 [Data Cleaning Pipeline](cleaning/README.md)

---
<div align="center">
  <b>Built with ❤️ by the Graduation Project Team</b>
  <br>
  <i>Empowering Egyptian Retail with Data Intelligence</i>
</div>
