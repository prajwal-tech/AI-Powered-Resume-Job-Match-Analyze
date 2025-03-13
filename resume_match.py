import re
import spacy
import json
import docx2txt
import pandas as pd
import streamlit as st
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]
    return list(set(keywords))

def compute_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity * 100, 2)

app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match_resume():
    data = request.json
    resume_text = data.get("resume_text", "")
    job_text = data.get("job_text", "")
    
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_text)
    similarity_score = compute_similarity(resume_text, job_text)
    
    return jsonify({
        "resume_keywords": resume_keywords,
        "job_keywords": job_keywords,
        "match_score": similarity_score
    })

# Streamlit UI
def main():
    st.title("AI-Powered Resume & Job Match Analyzer")
    uploaded_resume = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    uploaded_job = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
    resume_text = ""
    job_text = ""
    
    if uploaded_resume:
        if uploaded_resume.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_resume)
        else:
            resume_text = extract_text_from_docx(uploaded_resume)
    
    if uploaded_job:
        if uploaded_job.name.endswith(".pdf"):
            job_text = extract_text_from_pdf(uploaded_job)
        else:
            job_text = extract_text_from_docx(uploaded_job)
    
    if st.button("Analyze Match"):
        if resume_text and job_text:
            resume_keywords = extract_keywords(resume_text)
            job_keywords = extract_keywords(job_text)
            similarity_score = compute_similarity(resume_text, job_text)
            
            st.subheader("Results:")
            st.write(f"Match Score: {similarity_score}%")
            st.write("**Resume Keywords:**", ', '.join(resume_keywords))
            st.write("**Job Keywords:**", ', '.join(job_keywords))
        else:
            st.warning("Please upload both resume and job description files.")

if __name__ == "__main__":
    main()
