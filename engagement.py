import random


def recruiter_message(job_description, matched_skills=None):
    
    jd_preview = job_description.strip()[:150]

    skills_line = ""
    if matched_skills:
        skills_line = f"We noticed your experience in {', '.join(matched_skills[:3])} aligns well with our requirements.\n\n"

    return f"""
Hello,

We came across your profile and found it relevant for the following role:

{jd_preview}...

{skills_line}We believe you could be a strong fit for this opportunity.

Would you be interested in discussing this further?

Best regards,  
Recruitment Team
"""


def simulate_candidate_response(
    category,
    match_score,
    job_description
):

    category = category.lower()

    jd = job_description.lower()

    # -----------------------------------
    # HIGH MATCH CANDIDATES
    # -----------------------------------

    if match_score >= 70:

        responses = [

            "This role strongly matches my background. I would love to discuss further.",

            "Yes, this opportunity sounds exciting and relevant to my experience.",

            "I am actively looking for opportunities like this.",

            "This aligns very well with my skills and interests."
        ]

    # -----------------------------------
    # MEDIUM MATCH
    # -----------------------------------

    elif match_score >= 40:

        responses = [

            "I would like to know more about this opportunity.",

            "Sounds interesting. Can you share more details?",

            "I may be interested depending on the role requirements.",

            "This opportunity looks somewhat aligned with my experience."
        ]

    # -----------------------------------
    # LOW MATCH
    # -----------------------------------

    else:

        responses = [

            "Currently exploring opportunities.",

            "Not sure if this role matches my interests.",

            "I may not be the best fit for this role.",

            "At the moment I am focused on different opportunities."
        ]

    # -----------------------------------
    # CATEGORY BOOST
    # -----------------------------------

    if "data" in category and (
        "machine learning" in jd
        or "data" in jd
    ):

        responses.append(

            "I am very interested in data and AI-related opportunities."
        )

    if "python" in category and (
        "python" in jd
    ):

        responses.append(

            "Python development roles are highly interesting to me."
        )

    return random.choice(responses)


def calculate_interest_score(response):

    response = response.lower()

    positive_words = [

        "love",
        "exciting",
        "interested",
        "aligned",
        "looking",
        "yes",
        "discuss",
        "opportunity"

    ]

    negative_words = [

        "not",
        "unsure",
        "different",
        "may not"
    ]

    score = 50

    for word in positive_words:

        if word in response:

            score += 8

    for word in negative_words:

        if word in response:

            score -= 10

    return max(
        min(score, 100),
        0
    )