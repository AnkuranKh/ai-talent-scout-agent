import re

# -----------------------------------
# VALID SKILLS (CURATED)
# -----------------------------------

VALID_SKILLS = set([
    "python", "sql", "java", "c++",
    "machine learning", "ml", "nlp",
    "tensorflow", "pytorch",
    "react", "javascript", "html", "css",
    "node", "node.js",
    "django", "flask", "api",
    "aws", "docker", "kubernetes",
    "pandas", "numpy", "data analysis",
    "etl", "spark",
    "seo", "marketing", "sales",
    "recruitment", "payroll",
    "accounting", "finance", "banking"
])

# -----------------------------------
# JD PARSER
# -----------------------------------

def parse_jd(jd_text):

    jd_lower = jd_text.lower()

    words = re.sub(r'[^\w\s]', '', jd_lower).split()

    stopwords = [
        "the", "and", "for", "with", "a", "an", "to",
        "of", "in", "on", "we", "are", "looking",
        "developer", "engineer", "experience",
        "skills", "skill", "required", "preferred",
        "role", "job", "candidate"
    ]

    detected = set()

    # -----------------------------------
    # Multi-word skill detection
    # -----------------------------------

    for skill in VALID_SKILLS:
        if skill in jd_lower:
            detected.add(skill)

    # -----------------------------------
    # Single-word skill detection
    # -----------------------------------

    for word in words:
        if word not in stopwords and word in VALID_SKILLS:
            detected.add(word)

    return {
        "keywords": list(detected)
    }