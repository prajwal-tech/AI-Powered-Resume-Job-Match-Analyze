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
    """Extracts important keywords from text using NLP."""
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]
    return list(set(keywords))

def compute_similarity(resume_text, job_text):
    """Computes similarity score between a resume and a job description."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity * 100, 2)

def generate_feedback(similarity_score, resume_keywords, job_keywords):
    """Generates feedback based on match score and keyword comparison."""
    missing_keywords = list(set(job_keywords) - set(resume_keywords))
    
    if similarity_score >= 80:
        feedback = "✅ Excellent match! Your resume aligns well with the job description."
    elif 60 <= similarity_score < 80:
        feedback = "👍 Good match, but you can improve by adding more relevant skills."
    elif 40 <= similarity_score < 60:
        feedback = "⚠️ Moderate match. Consider tweaking your resume to better match the job description."
    else:
        feedback = "❌ Low match. Try tailoring your resume with relevant skills & experience."

    if missing_keywords:
        feedback += f"\n🔍 **Consider adding these skills to improve your match:** {', '.join(missing_keywords)}"
    
    return feedback

# Streamlit UI
def main():
    st.title("🚀 AI-Powered Resume & Job Match Analyzer")
    
    # Allow multiple resume uploads
    uploaded_resumes = st.file_uploader("📂 Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    uploaded_job = st.file_uploader("📂 Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    
    if uploaded_resumes and uploaded_job:
        # Extract job description text
        if uploaded_job.name.endswith(".pdf"):
            job_text = extract_text_from_pdf(uploaded_job)
        else:
            job_text = extract_text_from_docx(uploaded_job)
        
        job_keywords = extract_keywords(job_text)
        results = []
        
        for resume in uploaded_resumes:
            if resume.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(resume)
            else:
                resume_text = extract_text_from_docx(resume)
            
            # Compute similarity
            similarity_score = compute_similarity(resume_text, job_text)
            resume_keywords = extract_keywords(resume_text)
            feedback = generate_feedback(similarity_score, resume_keywords, job_keywords)
            
            results.append({
                "Resume": resume.name,
                "Match Score (%)": similarity_score,
                "Feedback": feedback
            })
        
        # Display results
        st.subheader("📊 Match Results & Feedback")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
if __name__ == "__main__":
    main()


