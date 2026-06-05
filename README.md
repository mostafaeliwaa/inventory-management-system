<div align="center">

# 📊 Retail Intelligence & Inventory Management System

**An End-to-End AI and Data Analytics Platform for Modern Retail**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

*Empowering 13 branches across Egypt with Machine Learning, Time-Series Forecasting, and Automated Data Pipelines.*

</div>

---

## 📖 About the Project

Retail inventory management is traditionally reactive. This system transforms it into a **proactive, predictive, and intelligent** operation. By leveraging robust machine learning models and deep analytics, this project identifies exactly what stock is needed, where it should be, and when it should be replenished.

**Core Capabilities:**
1. **Demand Forecasting:** Anticipating weekly and daily revenue streams to streamline cash flow.
2. **Inventory Optimization:** Classifying products dynamically to eliminate dead stock.
3. **Automated Invoicing (OCR):** Digitizing paper trails instantly with AI.
4. **Conversational AI:** Answering complex stock queries via a LangGraph-powered assistant.

---

## ⚡ Key Modules

### 1. 🧠 Machine Learning Engine (`/ML models/`)
- **Demand Classifier:** Categorizes items into Fast, Medium, or Slow moving to optimize supply chain routing. *(Random Forest, F1: 0.90)*
- **Dead-Stock Detector:** Accurately flags inventory at risk of becoming obsolete, saving thousands in holding costs. *(Gradient Boosting, ROC-AUC: 0.998)*
- **Interactive UI:** A dedicated Streamlit interface on port `8501` for on-the-fly predictions.

### 2. 📈 Time-Series Forecasting (`/forecasting/`)
- Predicts revenue across 8 key branches utilizing 144 days of granular POS data.
- **Ensemble Models:** Combines Prophet and LightGBM to gracefully handle holiday shocks (Ramadan/Eid) and weekly trends.
- **Accuracy:** Achieves an impressive WAPE of `0.229` for 4-week horizon planning.

### 3. 🧾 Smart OCR & Product Matching (`/ocr1/`)
- Powered by Google Gemini and Supabase.
- Automatically extracts line items from physical invoices.
- Matches extracted items against a master product database (2,400+ SKUs) using fuzzy logic.

### 4. 📊 Data Analytics (`/analysis/`)
- Deep-dive Jupyter notebooks covering ABC Classification and Anomaly Detection.
- Generates interactive Plotly dashboards for executive-level reporting.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & APIs** | `FastAPI`, `Uvicorn`, `Pydantic` |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-Learn`, `LightGBM`, `Prophet` |
| **Generative AI & OCR**| `LangGraph`, `Google Gemini API` |
| **Database** | `Supabase` (PostgreSQL) |
| **Frontend** | `Streamlit`, `Plotly` |

---

## 🚀 Getting Started

### Prerequisites
- **Python:** 3.8 or higher.
- **Environment:** Supabase URL/Key and Gemini API Key configured in your `.env` file.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mostafaeliwaa/inventory-management-system.git
cd inventory-management-system

# 2. Install global requirements
pip install -r requirements.txt
```

### Running the Services

**A. Start the Unified Backend (API Gateway)**
Serves OCR, Forecasting, and the AI Chatbot.
```bash
uvicorn main:app --port 8000 --reload
# View the API docs at http://localhost:8000/docs
```

**B. Start the Machine Learning UI**
Interact with the Demand Classifier and Dead-Stock predictor.
```bash
cd "ML models"
pip install -r requirements.txt
streamlit run streamlit_app.py
# View the UI at http://localhost:8501
```

---

## 📁 Repository Map

```text
inventory-management-system/
├── analysis/          # Exploratory Data Analysis & HTML Dashboards
├── chatbot/           # LangGraph conversational agent
├── cleaning/          # Data preprocessing pipelines
├── data/              # Datasets and schemas
├── forecasting/       # Time-Series prediction models & API
├── ML models/         # Classification Streamlit app & models
├── ocr1/              # Invoice digitization pipeline
├── routers/           # FastAPI sub-routers
└── main.py            # Unified API server
```

---

## 🤝 Documentation & Contribution
For deeper technical dives, consult the module-specific documentation:
- 📖 [System Architecture (CLAUDE.md)](CLAUDE.md)
- 📊 [Analytics Engine](analysis/README.md)
- 🧠 [Machine Learning Pipeline](ML%20models/README.md)
- 📈 [Forecasting Module](forecasting/README.md)

<div align="center">
<br>
<i>Engineered with precision to elevate Egyptian Retail.</i>
</div>
