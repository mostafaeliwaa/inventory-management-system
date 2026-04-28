
---

#  AI-Powered Inventory Intelligence System

An advanced **Autonomous Agent** built with **LangGraph** and **Google Gemini** to interact seamlessly with inventory databases. This assistant doesn't just chat; it reasons, generates real-time SQL queries, and provides data-driven insights about warehouse stock.


---

##  System Architecture

The system follows a **Modular Agentic Workflow**, separating concerns into distinct layers for scalability and reliability:

1.  **Orchestration Layer (LangGraph):** Manages the state machine and decision-making flow between the LLM and database tools.
2.  **Reasoning Engine (Google Gemini):** Analyzes natural language questions and determines if a database lookup is required.
3.  **Data Access Layer (Supabase):** Securely executes read-only SQL queries via a protected RPC (Remote Procedure Call) layer.
4.  **Interface Layer:** Offers both an interactive CLI for testing and a structured API for production integration.

---

##  Key Features

* **Natural Language to SQL:** Converts complex Arabic/English questions into optimized SQL queries.
* **Safe Data Access:** Strictly enforces `SELECT` permissions to ensure data integrity.
* **Stateful Conversations:** Tracks conversation context using LangGraph's state management.
* **Error Resilient:** Handles SQL errors and "No Results" scenarios gracefully.
* **Production Ready:** Built-in support for LangGraph Studio and environment-based configuration.

---

##  Project Structure

```text
├── Agents/               # Graph orchestration and node definitions
├── DB_Management/        # Supabase client and query execution logic
├── models/               # LLM configurations and prompt templates
├── Tools/                # SQL execution tools and base abstractions
├── config/               # Environment variables and settings management
├── AI_Inventory_Assistant_app.py  # Production Entry Point
└── main.py               # Interactive CLI for testing
```

---

##  Tech Stack

* **Framework:** LangGraph / LangChain
* **LLM:** Google Gemini (Generative AI)
* **Database:** Supabase (PostgreSQL)
* **Environment:** Python 3.12+
* **Dev Tools:** LangGraph Studio, Pydantic Settings

---

## System in Action (Output)


 ![alt text](image.png)




 ![alt text](image-1.png)

---

##  Setup & Installation

### 1. Environment Preparation
```bash
# Clone the repository
cd inventory-management-system/chatbot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Security Configuration
For the system to work, you must create the following function in your Supabase SQL Editor:
```sql
CREATE OR REPLACE FUNCTION execute_select(query text)
RETURNS json AS $$
BEGIN
  IF lower(trim(query)) NOT LIKE 'select%' THEN
    RAISE EXCEPTION 'Only SELECT queries are allowed for security reasons.';
  END IF;
  RETURN (EXECUTE format('SELECT json_agg(t) FROM (%s) t', query));
END;
$$ LANGUAGE plpgsql;
```

### 3. Run the System
* **Interactive CLI:** `python main.py`
* **LangGraph Studio:** `langgraph dev`

---
