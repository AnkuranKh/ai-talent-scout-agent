from engagement import (
    recruiter_message,
    simulate_candidate_response,
    calculate_interest_score
)
from data_loader import load_resumes
from jd_parser import parse_jd
import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="AI Talent Scout Agent",
    layout="wide"
)

@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------------
# FOR FAST RELOAD OF TURNS
# -----------------------------------
@st.cache(allow_output_mutation=True)
def compute_embeddings(candidates):
    for c in candidates:
        resume_text = c["resume"].strip().lower()
        c["embedding"] = model.encode([resume_text])
    return candidates



st.title("AI-Powered Talent Scouting Agent")

st.write("Find and engage the most relevant candidates using AI.")


# -----------------------------------
# CATEGORY DETECTION 
# -----------------------------------

from collections import defaultdict

# -----------------------------------
# KEYWORD MAP 
# -----------------------------------

category_keywords = {

    "CHEF": [
        "chef", "cooking", "kitchen", "cuisine", "food", "menu",
        "restaurant", "culinary", "recipe", "food preparation", "bakery"
    ],

    "ACCOUNTANT": [
        "accounting", "ledger", "tally", "gst", "tax", "taxation",
        "financial reporting", "audit", "bookkeeping", "balance sheet"
    ],

    "ADVOCATE": [
        "law", "legal", "court", "litigation", "advocate", "legal drafting",
        "case", "judicial", "contract law", "corporate law", "arbitration"
    ],

    "AGRICULTURE": [
        "farming", "agriculture", "crop", "soil", "irrigation",
        "fertilizer", "harvest", "agronomy", "pesticide"
    ],

    "APPAREL": [
        "garment", "textile", "fashion", "apparel", "fabric",
        "stitching", "pattern making", "clothing", "merchandising"
    ],

    "ARTS": [
        "art", "painting", "sketch", "illustration", "creative",
        "design", "visual art", "canvas", "drawing"
    ],

    "AUTOMOBILE": [
        "automobile", "vehicle", "engine", "car", "mechanic",
        "diagnostics", "automotive", "repair", "maintenance"
    ],

    "AVIATION": [
        "aviation", "aircraft", "flight", "pilot", "airline",
        "airport", "aeronautics", "aircraft systems"
    ],

    "BANKING": [
        "bank", "loan", "credit", "debit", "customer account",
        "financial services", "banking", "interest rate", "transaction"
    ],

    "BPO": [
        "bpo", "call center", "customer support", "voice process",
        "non voice", "client handling", "crm", "service desk"
    ],

    "BUSINESS-DEVELOPMENT": [
        "sales", "lead generation", "client acquisition", "business development",
        "revenue", "pipeline", "negotiation", "b2b", "b2c"
    ],

    "CONSTRUCTION": [
        "construction", "site", "civil", "project management",
        "building", "infrastructure", "contractor", "structural"
    ],

    "CONSULTANT": [
    "consulting", "strategy", "advisory", "business consulting"
    ],     

    "DESIGNER": [
    "ui", "ux", "figma", "photoshop", "illustrator",
    "wireframe", "prototyping", "graphic design"
    ],

    "DIGITAL MEDIA": [
        "social media", "content", "seo", "digital marketing",
        "campaign", "ads", "google analytics", "video editing"
    ],

    "ENGINEERING": [
        "engineering", "mechanical", "electrical", "civil",
        "technical", "design", "system", "core engineering"
    ],

    "FINANCE": [
        "finance", "investment", "portfolio", "stock", "market",
        "financial analysis", "risk", "budgeting", "forecasting"
    ],

    "FITNESS": [
        "fitness", "gym", "workout", "exercise", "trainer",
        "nutrition", "health coaching", "cardio", "strength training"
    ],

    "HEALTHCARE": [
        "medical", "healthcare", "patient", "hospital",
        "clinical", "treatment", "doctor", "nurse", "diagnosis"
    ],

    "HR": [
        "hr", "recruitment", "hiring", "talent", "payroll",
        "employee", "onboarding", "performance management"
    ],

    "INFORMATION TECHNOLOGY": [
    "software", "developer", "programming", "coding",
    "python", "java", "api", "backend", "frontend",
    "database", "sql", "nosql", "system design",
    "microservices", "cloud", "aws", "docker",
    "kubernetes", "server", "application", "rest api"
    ],

    "PUBLIC RELATIONS": [
        "pr", "media", "branding", "communication",
        "public relations", "press", "reputation management"
    ],

    "SALES": [
        "sales", "target", "revenue", "client", "closing",
        "deal", "pipeline", "conversion", "lead"
    ],

    "TEACHER": [
        "teaching", "education", "student", "classroom",
        "lesson", "curriculum", "school", "learning"
    ]
}

@st.cache(allow_output_mutation=True)
def build_category_embeddings():

    category_texts = defaultdict(list)

    # load ALL resumes
    candidates = load_resumes(None, filter_first=False)
    
    # group resumes by category
    for c in candidates:
        cat = str(c["category"]).strip().upper()
        category_texts[cat].append(c["resume"][:1000])  # limit size

    category_embeddings = {}

    for cat, texts in category_texts.items():

        # combine few resumes to represent category
        combined_text = " ".join(texts[:10])

        category_embeddings[cat] = model.encode([combined_text])

    return category_embeddings


def detect_category(jd):
    
    jd_lower = jd.strip().lower()
    jd_embedding = model.encode([jd_lower])
    category_embeddings = build_category_embeddings()

    scores = []

    for cat, emb in category_embeddings.items():
        score = cosine_similarity(jd_embedding, emb)[0][0]

        if cat in category_keywords:
            keyword_hits = sum(1 for kw in category_keywords[cat] if kw in jd_lower)

            # -----------------------------------
            # CONDITIONAL BOOSTING 
            # -----------------------------------
            if keyword_hits >= 3:
                score += 0.2
            elif keyword_hits == 2:
                score += 0.1
                
        scores.append((cat, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # -----------------------------------
    # FORCING INFORMATION TECHNOLOGY DUE TO AMBIGUITY
    # -----------------------------------
    it_keywords = category_keywords["INFORMATION TECHNOLOGY"]
    it_hits = sum(1 for kw in it_keywords if kw in jd_lower)

    if it_hits >= 3:
      return ["INFORMATION TECHNOLOGY"]

    top1, top2 = scores[0], scores[1]


    best_cat, best_score = top1
    second_cat, second_score = top2

    gap = best_score - second_score

    if gap < 0.10 and (second_score / best_score) > 0.90:
        return [best_cat, second_cat]
    else:
        return [best_cat]

# -----------------------------------
# SUBDOMAIN DETECTION (SEMANTIC)
# -----------------------------------

def detect_subdomain(category, jd_text):

    subdomain_map = {

        "INFORMATION TECHNOLOGY": [
            "machine learning and ai",
            "backend development apis",
            "frontend development ui ux",
            "devops cloud infrastructure"
        ],

        "DATA SCIENCE": [
            "machine learning models",
            "data analysis and analytics",
            "data engineering pipelines"
        ],

        "HR": [
            "recruitment hiring talent acquisition",
            "payroll and compensation",
            "training and development"
        ],

        "FINANCE": [
            "accounting and taxation",
            "investment and portfolio management",
            "banking and loans"
        ],

        "BUSINESS-DEVELOPMENT": [
            "sales and client acquisition",
            "marketing and seo",
            "operations and management"
        ]
    }

    valid_categories = [c for c in category if c in subdomain_map]

    if not valid_categories:
       return []

    jd_embedding = model.encode([jd_text.lower()])

    subdomains = list(set(
        sub for cat in valid_categories for sub in subdomain_map[cat]
    )
                      )

    scored = []

    for sub in subdomains:
        sub_embedding = model.encode([sub])
        score = cosine_similarity(jd_embedding, sub_embedding)[0][0]
        scored.append((sub, score))

    # sort by similarity
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    # pick top 2 subdomains
    return [s[0] for s in scored[:2] if s[1] > 0.3]


# -----------------------------------
# AI EXPLANATION GENERATOR
# -----------------------------------

def generate_rank_explanation(current, next_candidate, rank):
    
    explanation = f"Rank #{rank} candidate is placed above Rank #{rank+1} because "

    if current["match_score"] > next_candidate["match_score"]:
        explanation += f"they have a higher match score ({current['match_score']}% vs {next_candidate['match_score']}%). "

    if current["interest_score"] > next_candidate["interest_score"]:
        explanation += f"They also show higher interest ({current['interest_score']}% vs {next_candidate['interest_score']}%). "

    if len(current["matched_skills"]) > len(next_candidate["matched_skills"]):
        explanation += "Additionally, they match more required skills. "

    return explanation.strip()


# -----------------------------------
# SHOW TEST JDs 
# -----------------------------------

st.markdown("### 🧪 Test with Sample Job Descriptions")

if st.checkbox("📄 Show Test Job Descriptions"):

    try:
        with open("test_jds.txt", "r") as f:
            test_data = f.read()

        st.text_area(
            "Sample Job Descriptions (Copy & Paste Below)",
            value=test_data,
            height=300
        )

    except FileNotFoundError:
        st.error("❌ test_jds.txt file not found. Make sure it's in the same folder as app.py.")


# -----------------------------------
# JOB DESCRIPTION INPUT 
# -----------------------------------

job_description = st.text_area(
    "Paste Job Description",
    height=400,
    placeholder="""
📌 Enter Job Description

Include: Domain, Required Skills, Experience  
(Optional: Preferred Skills)

Example: Python backend developer with API & SQL experience.

💡 Use sample JDs above for testing.
"""
)


# -----------------------------------
# MAIN LOGIC
# -----------------------------------

if st.button("Find Best Candidates"):

    if not job_description.strip():
        st.warning("Please enter a job description.")

    else:

        parsed = parse_jd(job_description)
        jd_skills = parsed["keywords"]

        st.subheader("Extracted Requirements")

        for skill in jd_skills:
            st.write(f"✓ {skill}")

        detected_category = detect_category(job_description)
        st.session_state["detected_category"] = detected_category
        st.subheader(f"Detected Category: {', '.join(detected_category)}")
        
        
    

        subdomains = detect_subdomain(detected_category, job_description)

        if subdomains:
            st.subheader("Detected Subdomains")

        for sd in subdomains:
            st.write(f"✓ {sd}")
        candidates = load_resumes(detected_category, filter_first=True)

        if not candidates:
            candidates = load_resumes(None, filter_first=False)
        candidates = compute_embeddings(candidates)
        if not candidates:
            st.error("No candidates found")
            st.stop()

        jd_text = job_description.strip().lower()

        jd_embedding = model.encode([jd_text])
        results = []

        for candidate in candidates:

            resume_text = candidate["resume"].strip().lower()
            jd_text = job_description.strip().lower()
            category = candidate["category"]
            pdf_path = candidate.get("pdf_path")

            lower_resume = resume_text.lower()

            matched_skills = [
                skill for skill in jd_skills if skill in lower_resume
            ]

            skill_score = min(len(matched_skills) * 15, 60)

            resume_embedding = candidate["embedding"]

            similarity = cosine_similarity(jd_embedding, resume_embedding)[0][0]
            semantic_score = round(similarity * 100, 2)

            match_score = round(
                (0.3 * skill_score) +
                (0.7 * semantic_score),
                1
            )
            
            # -----------------------------------
            # CATEGORY DOMINANCE CHECK 
            # -----------------------------------

            category_counts = {}

            for cat, keywords in category_keywords.items():
                count = sum(1 for kw in keywords if kw in resume_text)
                category_counts[cat] = count

        
            
            # -----------------------------------
            # CATEGORY PENALTY 
            # -----------------------------------

            candidate_cat = category.strip().upper()

            if candidate_cat in detected_category:
                match_score += 5   # boost if matches ANY detected category
            else:
                match_score *= 0.7 # penalize if not in detected categories

            # re-round after adjustment
            match_score = round(match_score, 1)         
            
                     
            if matched_skills:
                ai_message = recruiter_message(job_description, matched_skills)
                candidate_response = simulate_candidate_response(category, match_score, job_description)
                interest_score = calculate_interest_score(candidate_response)
            else:
                ai_message = "Not contacted"
                candidate_response = "No response"
                interest_score = 0

            final_score = round((match_score * 0.7) + (interest_score * 0.3), 2)

            results.append({
                "id": candidate["id"],
                "category": category,
                "match_score": match_score,
                "interest_score": interest_score,
                "final_score": final_score,
                "matched_skills": matched_skills,
                "resume": resume_text,
                "ai_message": ai_message,
                "candidate_response": candidate_response,
                "pdf_path": pdf_path
            })

        results = sorted(
                results,
                key=lambda x: (x["final_score"], x["id"]),
                reverse=True
)

        threshold = 3  

        for i in range(len(results) - 1):
            curr = results[i]
            next_cand = results[i + 1]

            if abs(curr["match_score"] - next_cand["match_score"]) < threshold:
                if next_cand["interest_score"] > curr["interest_score"]:
                    results[i], results[i + 1] = results[i + 1], results[i]

        st.session_state["results"] = results


# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

results = st.session_state.get("results", [])

if results:

    st.subheader("Top Candidates")

    top_candidates = results[:3]

    for i in range(len(top_candidates)):

        r = top_candidates[i]
        rank = i + 1

        st.markdown("---")

        col1, col2 = st.columns([3, 1])

        with col1:

            st.markdown(f"## Rank #{rank}")
            st.write(f"Category: {r['category']}")

            if i < len(top_candidates) - 1:
                explanation = generate_rank_explanation(
                    r,
                    top_candidates[i + 1],
                    rank
                )
            else:
                explanation = "This candidate ranks lower compared to the top candidates based on overall scores."

            st.write("### Why Ranked")
            st.write(explanation)

            if r["matched_skills"]:
                st.write("**Matched Skills:**")
                for skill in r["matched_skills"]:
                    st.write(f"✓ {skill}")

            st.write("### AI Outreach")
            st.info(r["ai_message"])

            st.write("### Candidate Response")
            st.success(r["candidate_response"])
            
            
            st.write("### Resume Preview")
            st.write(r["resume"][:500] + "...")

            st.write("### Resume (PDF)")

            pdf_path = r.get("pdf_path")

            if pdf_path and os.path.isfile(pdf_path):

                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="⬇ Download Resume",
                    data=pdf_bytes,
                    file_name=f"{r['id']}.pdf",
                    mime="application/pdf"
                )

        with col2:

            st.metric("Match Score", f"{r['match_score']}%")
            st.metric("Interest Score", f"{r['interest_score']}%")
            st.metric("Final Score", f"{r['final_score']}%")

    st.success("Ranking completed.")
    
# -----------------------------------
# POOR CANDIDATES TABLE
# -----------------------------------

poor_candidates = [
    r for r in results
    if (
        20 <= r["match_score"] < 40
        and r["category"].strip().upper() in st.session_state.get("detected_category", [])
    )
]

if poor_candidates:

    st.subheader("🔍 Potential Candidates to Upskill")
    st.write("""
These candidates match the job category but are missing some important skills or experience. 
They may not be the best immediate fit but could be considered for future roles or upskilling.
""")

    table_data = []

    for r in poor_candidates:
        table_data.append({
            "Candidate ID": r["id"],
            "Category": r["category"],
            "Match Score": r["match_score"],
            "Interest Score": r["interest_score"],
            "Matched Skills": ", ".join(r["matched_skills"]) if r["matched_skills"] else "None"
        })

    st.table(table_data)

    # -----------------------------------
    # SAFE PDF ACCESS (FIXED GROUPING)
    # -----------------------------------

    st.markdown("### 📄 View / Download Candidate Resumes")

    candidate_ids = [r["id"] for r in poor_candidates]

    selected_id = st.selectbox(
        "Select Candidate ID",
        candidate_ids,
        index=0  # ensures default selection
    )

    selected_candidate = next(
        (r for r in poor_candidates if r["id"] == selected_id),
        None
    )

    if selected_candidate:
        pdf_path = selected_candidate.get("pdf_path")

        if pdf_path and os.path.isfile(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label=f"⬇ Download Resume - {selected_id}",
                data=pdf_bytes,
                file_name=f"{selected_id}.pdf",
                mime="application/pdf"
            )