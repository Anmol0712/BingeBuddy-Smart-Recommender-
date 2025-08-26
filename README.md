# 🎬 BingeBuddy – Smart Content Recommendation System  

BingeBuddy is a **content-based recommendation system** that helps users discover movies and shows based on their preferences. It combines **text similarity (TF-IDF + Cosine Similarity)** and **ratings (SVD-reduced features + hybrid scoring)** to provide accurate and personalized recommendations.  

---

## 🚀 Features  
- 📑 **TF-IDF Vectorization** – Extracts meaningful features from content metadata.  
- 🔎 **SVD (Singular Value Decomposition)** – Reduces dimensionality for efficient recommendations.  
- 🤝 **Hybrid Scoring** – Balances similarity and user ratings for better accuracy.  
- 🎨 **Interactive Streamlit UI** – Clean and responsive interface for real-time recommendations.  
- ⚡ **Fast & Scalable** – Handles large content databases with optimized similarity search.  

---

## 🛠️ Tech Stack  
- **Python** (Pandas, NumPy, Scikit-learn)  
- **Streamlit** (Frontend UI)  
- **Joblib** (Model persistence)  
- **Cosine Similarity** (Recommendation logic)  

---

## 📂 Project Structure  
BingeBuddy/
│── app.py # Streamlit application
│── tfidf_vectorizer.pkl # Saved TF-IDF model
│── svd_model.pkl # Trained SVD model
│── tfidf_reduced.npy # Reduced TF-IDF features
│── content_metadata.pkl # Metadata dataset
│── requirements.txt # Dependencies
│── README.md # Project documentation
