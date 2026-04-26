# 🤖 AI-Powered Talent Scouting Agent

An intelligent recruitment assistant that automates candidate discovery, matching, engagement, and ranking using AI.
Dataset taken from Kaggle
---

# 🚀 Overview

Recruiters spend hours manually screening resumes and reaching out to candidates.
This project solves that by building an **AI agent** that:

* Understands a Job Description (JD)
* Finds relevant candidates
* Engages them with AI-generated messages
* Evaluates their interest
* Produces a ranked shortlist

---

# ⚡ Quick Start (Run Locally)

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-talent-scout-agent.git

cd ai-talent-scout-agent

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

# ⚠️ Important Notes

* Ensure **Python 3.9+** is installed
* Dataset (`Resume.csv`) and resume PDFs are included
* First run may take time (model loading)

---

# 🎯 Key Features

### 1. 📄 Job Description Parsing

* Extracts **relevant skills** from raw JD text
* Uses curated skill matching with NLP preprocessing

---

### 2. 🧠 Intelligent Category Detection

A hybrid system combining:

* **Semantic embeddings (Sentence Transformers)**
* **Keyword-based boosting**
* **Rule-based corrections (for edge cases like IT vs Engineering)**

---

### 3. 🔍 Candidate Matching Engine

Each candidate is scored using:

* **Skill Match Score** → based on overlapping skills
* **Semantic Score** → cosine similarity between JD & resume embeddings

Final Match Score:

```
Match Score = 0.3 × Skill Score + 0.7 × Semantic Score
```

---

### 4. 💬 AI Candidate Engagement

* Generates **personalized recruiter messages**
* Simulates **candidate responses**
* Calculates an **Interest Score** using sentiment-based logic

---

### 5. 🏆 Smart Ranking System

Final ranking is based on:

```
Final Score = 0.7 × Match Score + 0.3 × Interest Score
```

Additional logic:

* Category alignment boost
* Tie-breaking using interest score
* Explainable ranking between candidates

---

### 6. 📊 Explainability

Each candidate includes:

* Why they are ranked higher
* Matched skills
* Engagement response

---

### 7. 📉 Upskilling Candidate Insights

A dedicated section shows:

* Candidates with moderate match (20–40%)
* Same category but missing skills
* Helps recruiters identify **trainable talent**

---

### 8. 📄 Resume Handling

* Resume preview inside UI
* Download full PDF
* Clean recruiter-friendly layout

---

# ⚡ Performance Optimizations

* Cached model loading using Streamlit
* Resume embeddings computed once (faster re-runs)
* Efficient filtering before ranking

---

# 🧪 Sample Workflow

1. Paste a Job Description
2. System extracts required skills
3. Detects job category & subdomain
4. Finds matching candidates
5. Engages candidates with AI
6. Ranks them based on match + interest
7. Displays top candidates + explanations

---

# 🏗️ Project Structure

```
├── app.py               # Main Streamlit application
├── data_loader.py       # Resume loading & filtering
├── jd_parser.py         # Skill extraction from JD
├── engagement.py        # AI outreach & response simulation
├── matcher.py           # (Legacy) TF-IDF approach
├── conversation.py      # (Experimental) interest simulation
├── Resume.csv           # Dataset
├── data/                # Resume PDFs
├── requirements.txt     # Dependencies
├── README.md
```

---

# 🔄 Evolution of the System

This project went through multiple iterations:

### ❌ Initial Approach

* TF-IDF based similarity
* No category filtering
* Static candidate responses

### 🚫 Limitations

* Poor semantic understanding
* Irrelevant candidates ranked higher
* No recruiter insight

### ✅ Final Approach

* Transformer-based embeddings
* Hybrid category detection
* Dynamic engagement simulation
* Explainable ranking system

---

# 🧠 Technologies Used

* **Python**
* **Streamlit**
* **Sentence Transformers**
* **Scikit-learn**
* **Pandas**

---

# 📌 Example Input

```
Domain: Information Technology  
Required Skills: Python, SQL, APIs  
Preferred Skills: AWS, Docker  
Experience: 2+ years
```

---

# 📈 Output

* Top 3 ranked candidates
* Match Score + Interest Score
* AI-generated explanation
* Resume preview + download
* Upskilling candidate suggestions

---

# 🎯 Use Cases

* Recruitment automation
* Talent discovery platforms
* HR tech tools
* Internal hiring systems

---

# 🔮 Future Improvements

* Real-time candidate communication
* Integration with LinkedIn / ATS
* LLM-based resume understanding
* Skill gap analysis with recommendations

---

# 🏁 Conclusion

This AI agent transforms recruitment from a manual process into an **intelligent, automated, and explainable system**—helping recruiters make faster and better hiring decisions.

---

# 👨‍💻 Author

Built as part of an AI hackathon project by Ankuran Khanikar
