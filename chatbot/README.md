# Chatbot Agent 

This chatbot agent responds to customer inquiries based on the client’s warehouse database, providing accurate information about inventory, product availability, and stock-related details.

## Requirements

- Python 3.10 - 3.12 (3.14 not supported due to missing pre-built wheels)

## Setup

### 1. Create Virtual Environment

```bash
cd chatbot

# Create venv with Python 3.12
python -3.12 -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

### 3.Testing using Studio

```bash
langgraph dev

open http://127.0.0.1:2024
```