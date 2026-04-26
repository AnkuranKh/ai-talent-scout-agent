import pandas as pd
import re
import os


# -----------------------------------
# NORMALIZATION FUNCTION
# -----------------------------------

def normalize(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


# -----------------------------------
# ABBREVIATION MAP
# -----------------------------------

abbreviation_map = {
    "it": "information technology",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "hr": "human resources",
    "bd": "business development",
    "ds": "data science",
    "fin": "finance"
}


# -----------------------------------
# MAIN LOADER
# -----------------------------------

def load_resumes(category=None, filter_first=True):

    df = pd.read_csv("Resume.csv")

    # -----------------------------------
    # FILTER BY CATEGORY
    # -----------------------------------

    if filter_first and category and category != "GENERAL":

        normalized_category = normalize(category)

        expanded_category = abbreviation_map.get(
            normalized_category,
            normalized_category
        )

        normalized_expanded = normalize(expanded_category)

        df = df[
            df["Category"]
            .fillna("")
            .apply(lambda x: normalize(x))
            .str.contains(normalized_expanded)
        ]

    # -----------------------------------
    # HANDLE EMPTY CASE
    # -----------------------------------

    if df.empty:
        return []

    # -----------------------------------
    # SAMPLE DATA
    # -----------------------------------

    #df = df.sample(min(len(df), 100)).reset_index(drop=True)

    # -----------------------------------
    # GET AVAILABLE FOLDERS
    # -----------------------------------

    base_path = "data"
    available_folders = os.listdir(base_path)

    # Normalize all folder names once
    folder_map = {
        normalize(folder): folder for folder in available_folders
    }

    # -----------------------------------
    # BUILD CANDIDATES
    # -----------------------------------

    candidates = []

    for _, row in df.iterrows():

        category_name = str(row["Category"]).strip()
        normalized_cat = normalize(category_name)

        # Match with actual folder
        matched_folder = folder_map.get(normalized_cat)

        pdf_path = None

        if matched_folder:
            pdf_path = os.path.join(base_path, matched_folder, f"{row['ID']}.pdf")

            if not os.path.isfile(pdf_path):
                pdf_path = None

        candidate = {
            "id": row["ID"],
            "resume": str(row["Resume_str"]),
            "category": category_name,
            "pdf_path": pdf_path
        }

        candidates.append(candidate)

    return candidates
# -----------------------------------
# GET ALL CATEGORIES (NEW FUNCTION)
# -----------------------------------

def get_all_categories():
    df = pd.read_csv("Resume.csv")
    return list(df["Category"].dropna().unique())