from sklearn.metrics.pairwise import cosine_similarity


def calculate_match(job_description,
                    resume_text):

    documents = [
        job_description,
        resume_text
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)