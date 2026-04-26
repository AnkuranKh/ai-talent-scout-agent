def simulate_interest(candidate):
    
    fake_response = "I am interested in this opportunity"

    positive_words = [
        "interested",
        "opportunity",
        "yes"
    ]

    score = 0

    for word in positive_words:

        if word in fake_response.lower():
            score += 30

    return min(score, 100)