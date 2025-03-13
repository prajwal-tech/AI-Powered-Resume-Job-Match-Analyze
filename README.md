AI-Powered Resume & Job Match Analyzer

Overview

This project is an AI-powered application that compares a resume with a job description to determine their match percentage. It extracts keywords and calculates similarity using Natural Language Processing (NLP) and Machine Learning techniques.

Features

✅ Resume & Job Description Parsing (Supports PDF & DOCX)

✅ NLP-Based Keyword Extraction (Using spaCy)

✅ Similarity Matching (TF-IDF + Cosine Similarity)

✅ Flask API (For programmatic access)

✅ Streamlit Web UI (For interactive use)

Installation

1. Clone the Repository

2.Install Dependencies

pip install -r requirements.txt

python -m spacy download en_core_web_sm

3. Run the Application

streamlit run resume_match.py

Usage

Upload a resume (PDF/DOCX) and a job description (PDF/DOCX).

Click "Analyze Match".

View the match percentage and extracted keywords.
