<div align="center">
  
# 📦 AI-Powered Retail Inventory Management System

**A comprehensive, data-driven solution for retail distribution with 13 branches across Egypt.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Machine_Learning-orange)](#)
[![Prophet](https://img.shields.io/badge/Prophet-Time_Series-blue)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas)](#)

*Empowering decision-makers with Interactive Dashboards, Machine Learning, and Time-Series Forecasting.*

</div>

---

## 🌟 Overview

This project is a robust, end-to-end **Inventory Management System** designed to tackle real-world retail challenges. By integrating advanced **Data Analytics**, **Machine Learning**, and **Time-Series Forecasting**, the system moves beyond traditional inventory tracking to provide actionable insights, optimize stock levels, and predict future revenue.

### 🎯 Key Objectives
- **Minimize Dead Stock:** Identify non-moving inventory before it eats into profit margins.
- **Optimize Distribution:** Classify product demand (Fast/Medium/Slow) to ensure the right products are at the right branches.
- **Forecast Revenue:** Accurately predict daily and weekly branch revenue to optimize staffing, cash flow, and logistics.

---

## 🚀 Features

### 📊 1. Advanced Analytics & Dashboards
- **ABC Classification:** Segment 2,400+ products by sales volume and revenue.
- **Branch Performance:** Compare KPIs across 13 retail branches.
- **Anomaly Detection:** Automatically flag unusual stock-to-sales ratios.
- **Interactive Visualizations:** Responsive HTML dashboards powered by Plotly.

### 🧠 2. Machine Learning Suite (`/ML models/`)
- **Demand Classification:** A Random Forest model predicting Fast/Medium/Slow moving items (Macro F1: ~0.90).
- **Dead-Stock Predictor:** A Gradient Boosting model identifying static inventory with extreme precision (ROC-AUC: 0.998).
- **Interactive UI:** Built-in Streamlit app (`port 8501`) for manual testing and batch inference.
- **RESTful API:** FastAPI service (`port 8003`) serving live predictions.

### 📈 3. Time-Series Forecasting (`/forecasting/`)
Built on 144 days of POS data across 8 branches, handling Ramadan/Eid shocks, weekly cycles, and payday/school-holiday windows.
- **Daily Forecasts (7-day horizon):** Prophet + LightGBM Ensemble (WAPE: 0.479).
- **Weekly Forecasts (4-week horizon):** Ensemble3 (Prophet + LGBM + Naive Mean) for robust cash-flow planning (WAPE: 0.229).
- **Forecasting API:** Dedicated FastAPI service (`port 8004`) for seamless integration.

### 🤖 4. AI Assistant & Future Integrations
- **Chatbot:** LangGraph-powered AI assistant (`port 8002`) for natural language querying.
- **Database:** Supabase integration for normalized, cloud-based data storage.
- **OCR Pipeline:** Processing of scanned invoices and receipts (`/ocr1/`).

---

## 🏗️ Architecture & Tech Stack

| Domain | Technologies Used |
| :--- | :--- |
| **Core & Data Prep** | `Python`, `Pandas`, `NumPy`, `Scikit-learn` |
| **Machine Learning** | `LightGBM`, `Random Forest`, `Gradient Boosting` |
| **Time Series** | `Prophet`, `LightGBM (Recursive & Direct)` |
| **APIs & Backend** | `FastAPI`, `Uvicorn`, `Pydantic`, `Supabase` |
| **Frontend & BI** | `Streamlit`, `Plotly`, `Jupyter Notebooks` |

---

## 🏁 Quick Start

### Prerequisites
Make sure you have Python 3.8+ installed. 

### 1. Clone & Install
```bash
git clone https://github.com/mostafaeliwaa/inventory-management-system.git
cd inventory-management-system
pip install -r requirements.txt
```

### 2. Launch the ML Streamlit Interface
```bash
cd "ML models"
pip install -r requirements.txt
streamlit run streamlit_app.py
# The UI will open at http://localhost:8501
```

### 3. Start the Forecasting API
```bash
cd forecasting
pip install -r requirements.txt
uvicorn api:app --port 8004 --reload
# Access the interactive API docs at http://localhost:8004/docs
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
 ┣ 📜 main.py          # Main application entry point
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
