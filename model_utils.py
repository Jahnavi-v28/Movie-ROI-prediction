
import joblib
import pandas as pd
from pathlib import Path


# ==========================================
# 1. Load Saved Model and Preprocessing Files
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

genre_encoder = joblib.load(MODEL_DIR / "genre_encoder.pkl")
keyword_encoder = joblib.load(MODEL_DIR / "keyword_encoder.pkl")
company_encoder = joblib.load(MODEL_DIR / "company_encoder.pkl")
country_encoder = joblib.load(MODEL_DIR / "country_encoder.pkl")
language_encoder = joblib.load(MODEL_DIR / "language_encoder.pkl")


print("Random Forest model loaded successfully.")
print(f"Expected feature count: {len(feature_columns)}")
print("Genre classes:", len(genre_encoder.classes_))
print("Keyword classes:", len(keyword_encoder.classes_))
print("Company classes:", len(company_encoder.classes_))
print("Country classes:", len(country_encoder.classes_))
print("Language classes:", len(language_encoder.classes_))


# ==========================================
# 2. Helper Function: Convert Input to a List
# ==========================================

def to_list(values):
    """Accept a list or comma-separated string."""
    if values is None:
        return []

    if isinstance(values, str):
        return [
            item.strip()
            for item in values.split(",")
            if item.strip()
        ]

    return [
        str(item).strip()
        for item in values
        if str(item).strip()
    ]


# ==========================================
# 3. Build Features Using Training Column Names
# ==========================================

def build_movie_features(
    budget,
    runtime,
    franchise,
    release_year,
    release_month,
    original_language,
    release_season,
    director,
    genres,
    keywords,
    companies,
    countries,
    languages,
    overview
):
    # Initialize every expected feature to zero.
    features = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    # --------------------------------------
    # Numeric features
    # --------------------------------------

    numeric_values = {
        "budget": float(budget),
        "runtime": float(runtime),
        "franchise": int(franchise),
        "release_year": int(release_year),
        "release_month": int(release_month),
    }

    for name, value in numeric_values.items():
        if name in features.columns:
            features.loc[0, name] = value

    # --------------------------------------
    # One-hot categorical features
    # Training used pd.get_dummies(drop_first=True)
    # --------------------------------------

    categorical_values = {
        "original_language": original_language,
        "release_season": release_season,
        "director_clean": director,
    }

    for column, value in categorical_values.items():
        if value is None:
            continue

        value = str(value).strip()
        feature_name = f"{column}_{value}"

        if feature_name in features.columns:
            features.loc[0, feature_name] = 1.0

    # --------------------------------------
    # Multi-label features
    # Match the prefixes used in Notebook 3.
    # Genres have no prefix; other groups do.
    # --------------------------------------

    encoder_inputs = [
    (genre_encoder, genres, "genre_list_"),
    (keyword_encoder, keywords, "keyword_list_"),
    (company_encoder, companies, "company_list_"),
    (country_encoder, countries, "country_list_"),
    (language_encoder, languages, "language_list_"),
]

    for encoder, values, prefix in encoder_inputs:
        values = to_list(values)

        encoded_values = encoder.transform([values])[0]

        for label, flag in zip(encoder.classes_, encoded_values):
            if prefix is None:
                feature_name = str(label)
            else:
                feature_name = f"{prefix}{label}"

            if feature_name in features.columns:
                features.loc[0, feature_name] = float(flag)

    # --------------------------------------
    # TF-IDF overview features
    # --------------------------------------

    overview = overview or ""
    overview_vector = tfidf.transform([overview]).toarray()[0]
    overview_names = tfidf.get_feature_names_out()

    for name, value in zip(overview_names, overview_vector):
        if name in features.columns:
            features.loc[0, name] = float(value)

    # --------------------------------------
    # Final feature alignment
    # --------------------------------------

    features = features.reindex(
        columns=feature_columns,
        fill_value=0.0
    )

    if features.shape[1] != len(feature_columns):
        raise ValueError("Feature count does not match the trained model.")

    return features


# ==========================================
# 4. Predict Movie ROI
# ==========================================

def predict_movie_roi(**movie_details):
    features = build_movie_features(**movie_details)

    prediction = float(model.predict(features)[0])

    return prediction


# ==========================================
# 5. Basic Validation
# ==========================================




if __name__ == "__main__":
    print("\nTesting movie ROI prediction...")

    sample_movie = {
        "budget": 20000000,
        "runtime": 120,
        "franchise": 0,
        "release_year": 2026,
        "release_month": 7,
        "original_language": "en",
        "release_season": "Summer",
        "director": "Other",
        "genres": ["Drama"],
        "keywords": ["friendship"],
        "companies": [],
        "countries": ["United States of America"],
        "languages": ["English"],
        "overview": "A family discovers a mysterious world."
    }

    try:
        predicted_roi = predict_movie_roi(**sample_movie)
        print(f"Predicted ROI: {predicted_roi:.4f}")
    except Exception as error:
        print("Prediction failed:", error)
