import re
import spacy
import json
import docx2txt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
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

# Streamlit UI
def main():
    st.title("AI-Powered Resume & Job Match Analyzer")
    
    # Allow multiple resume uploads
    uploaded_resumes = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    uploaded_job = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
    if uploaded_resumes and uploaded_job:
        # Extract job description text
        if uploaded_job.name.endswith(".pdf"):
            job_text = extract_text_from_pdf(uploaded_job)
        else:
            job_text = extract_text_from_docx(uploaded_job)
        
        results = []
        
        for resume in uploaded_resumes:
            if resume.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(resume)
            else:
                resume_text = extract_text_from_docx(resume)
            
            # Compute similarity
            similarity_score = compute_similarity(resume_text, job_text)
            resume_keywords = extract_keywords(resume_text)
            
            results.append({
                "Resume": resume.name,
                "Match Score": similarity_score,
                "Keywords": ", ".join(resume_keywords)
            })
        
        # Display results
        st.subheader("Match Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
if __name__ == "__main__":
    main()

