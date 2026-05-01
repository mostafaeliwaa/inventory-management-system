# 📄 OCR - Invoice Processing & Product Matching

## Overview

The OCR (Optical Character Recognition) feature automatically extracts data from invoice images and matches products with your database. It uses Google's Gemini AI for text extraction and a hybrid matching system (semantic + fuzzy) to identify products with high accuracy, even with Arabic/English mixed text and OCR noise.

## 🚀 Features

### 1. **Automatic Invoice Data Extraction**
- Extracts store/vendor name, invoice date, invoice number, and total amount
- Supports both Arabic and English invoices
- Handles mixed-script text (Arabic + English)
- Converts Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to Western digits (0123456789)

### 2. **Product Matching System**
- **Hybrid matching**: Combines semantic similarity (embeddings) + fuzzy matching
- **Multi-language support**: Optimized for Egyptian market (Arabic + English)
- **Text normalization**: Handles OCR noise, diacritics, and character variations
- **Learning capability**: Remembers user corrections for future matches

### 3. **Database Integration**
- **Supabase support**: Direct integration with Supabase (product database)
- **CSV fallback**: Automatically falls back to CSV if Supabase unavailable
- **Real-time updates**: New products added through corrections are saved

### 4. **API Endpoints**
- `/process-invoice`: Upload and process invoice images
- `/confirm-items`: Submit user corrections to improve matching
- `/health`: Check API and dependency status
- `/stats`: View matcher statistics

### 5. **Streamlit Dashboard**
- Upload invoices via drag-and-drop
- View extracted data (invoice metadata + products)
- Interactive product matching with editable fields
- Export results as JSON or CSV
- Processing history
