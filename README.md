# 💼 SmartHire – AI-Powered Job Recommendation System

SmartHire is an AI-powered job recommendation system that analyzes a candidate's resume and recommends relevant job opportunities using Natural Language Processing (NLP) and Machine Learning.

The system uses **TF-IDF vectorization and Cosine Similarity** to compare resume content with job descriptions. It also applies an explainable smart scoring mechanism based on skills, experience, location, and salary.

---

## 🚀 Features

- 📄 Upload a resume in DOCX format
- 🧠 Automatic resume text extraction
- 🔍 TF-IDF based text representation
- 📊 Cosine Similarity based job matching
- 🎯 Explainable Smart Match Score
- 💼 Skills-based matching
- 📈 Experience compatibility scoring
- 📍 Location matching
- 💰 Salary compatibility matching
- 🔎 Job filters
- 🏆 Top job recommendations
- 📋 Detailed match explanations
- 📥 Export recommendations as CSV
- 🌐 Streamlit web application

---

## 🧠 How SmartHire Works

The recommendation pipeline works as follows:

```text
Resume Upload
      ↓
DOCX Text Extraction
      ↓
TF-IDF Vectorization
      ↓
Job Filtering
      ↓
Cosine Similarity
      ↓
Top Candidate Jobs
      ↓
Explainable Smart Scoring
      ↓
Ranked Job Recommendations
