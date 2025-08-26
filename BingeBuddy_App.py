import streamlit as st
import base64
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

tfidf = joblib.load("tfidf_vectorizer.pkl")
svd = joblib.load("svd_model.pkl")
tfidf_reduced = np.load("tfidf_reduced.npy")
df = pd.read_pickle("content_metadata.pkl")

# --- Function to get recommendations ---
def get_recommendations(title, topN=10, alpha=0.8):
    """
    alpha = weight for similarity (0-1)
    (1-alpha) = weight for rating
    """
    # Find index of the movie/show
    idx = df[df["content_title"] == title].index[0]

    # Get the vector of the target movie
    target_vec = tfidf_reduced[idx].reshape(1, -1)

    # Compute cosine similarity
    sim_scores = cosine_similarity(target_vec, tfidf_reduced).flatten()

    # Compute hybrid score
    hybrid_scores = alpha * sim_scores + (1 - alpha) * df["avg_rating"].values

    # Sort scores (excluding itself)
    sim_indices = np.argsort(hybrid_scores)[::-1]
    sim_indices = [i for i in sim_indices if i != idx][:topN]

    # Build DataFrame with only required details
    results = df.loc[sim_indices, ["content_title", "content_genres", "content_type", "avg_rating"]].copy()
    results = results.reset_index(drop=True)

    return results




# ------------------- APP CONFIG -------------------
st.set_page_config(page_title="BingeBuddy", page_icon="🍿", layout="centered")

# ------------------- BACKGROUND + STYLES -------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: black;
        }
        body {
            background-color: #0d0d0d;
            color: white;
        }
        .content-box {
            background-color: rgba(26,26,26,0.7);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(255, 0, 0, 0.4);
            text-align: center;
            max-width: 700px;
            margin: auto;
        }
        .recommend-box {
            background-color: rgba(26,26,26,0.7);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- SHOW GIF -------------------
file_ = open("C:/Users/anmol/Documents/Data Science with Python(STP)/bingebuddy project/bingebuddy_gifs.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

# ------------ FULLSCREEN BACKGROUND ------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(to right, rgba(0,0,0,0.95), rgba(0,0,0,0.3), rgba(0,0,0,0.95)), 
            url("data:image/gif;base64,{data_url}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- HEADER -------------------
with st.container():
    st.markdown(
        """
        <div class="content-box">
            <h1 style="color:#e50914; font-weight:900; font-size:42px; margin-bottom:10px;">🍿 BingeBuddy</h1>
            <h3 style="color:white; font-size:20px; font-weight:400;">
                🎬 Your smart companion for endless entertainment picks 📺✨
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------- INPUT -------------------
st.write("")
user_input = st.text_input(
    "✨ Enter Movies / Anime / TV-Series / K-dramas you liked:",
    placeholder="e.g. Interstellar, Naruto, Squid Game"
)

# ------------------- BUTTONS -------------------
st.markdown("""
<style>
div.stButton > button {
    background: linear-gradient(135deg, #e50914, #b81d24);
    color: white !important;
    border: none;
    border-radius: 10px;
    width: 200px; /* same width for all buttons */
    height: 50px;
    font-weight: 600;
    font-size: 15px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    cursor: pointer;
}
            
div.stButton > button:hover {
    background: linear-gradient(135deg, #b81d24, #8c0d13);
    transform: translateY(-2px);
    box-shadow: 0px 6px 12px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,4,1])

with col2:
    c1,c2 = st.columns(2)
    with c1:
        similar_btn = st.button("🎬 Get Similar Recommendations")
    with c2:
        random_btn = st.button("🎲 Random Pick!")

# ------------------- RECOMMENDATION LOGIC -------------------
from difflib import get_close_matches
import re

def clean_title(text):
    """Lowercase + remove special characters for flexible matching"""
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower()).strip()

def find_best_match(user_input, df_titles):
    user_clean = clean_title(user_input)
    cleaned_titles = [clean_title(t) for t in df_titles]

    # Fuzzy match on cleaned titles
    matches = get_close_matches(user_clean, cleaned_titles, n=1, cutoff=0.6)
    if matches:
        idx = cleaned_titles.index(matches[0])
        return df_titles[idx]  # return original title
    return None


if similar_btn:
    if not user_input.strip():  # Warning if empty
        st.warning("⚠️ Please enter at least one title before getting recommendations.")
    else:
        try:
            # Find closest match instead of exact
            matched_title = find_best_match(user_input.strip(), df["content_title"].tolist())

            if not matched_title:
                st.error("❌ This title is not in our database. Try another one!")
            else:
                st.success(f"🔎 Found closest match: **{matched_title}**")
                recs = get_recommendations(matched_title, topN=20, alpha=0.8)

                st.subheader(f"🔥 Recommended for You (based on **{matched_title}**):")
                
                for _, row in recs.iterrows():
                    st.markdown(
                        f"""
                        <div class='recommend-box'>
                            🎥 <b>{row['content_title']}</b><br>
                            📌 {row['content_genres']} | 🎭 {row['content_type']} | ⭐ {row['avg_rating']*10:.1f}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

elif random_btn:
    st.subheader("🎲 Your Random Pick:")
    rec = df.sample(1).iloc[0]  # random from dataset instead of static list
    st.markdown(
        f"""
        <div class='recommend-box'>
            ✨ <b>{rec['content_title']}</b><br>
            📌 {rec['content_genres']} | 🎭 {rec['content_type']} | ⭐ {rec['avg_rating']*10:.1f}
        </div>
        """,
        unsafe_allow_html=True
    )