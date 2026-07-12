import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


rec = load_model()

st.title("🎬 Hybrid Movie Recommender")
st.caption(
    "Content-based (genre similarity) + collaborative filtering (rating patterns via SVD), "
    "trained on the MovieLens 1M dataset."
)

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    content_weight = st.slider(
        "Content vs. Collaborative weight", 0.0, 1.0, 0.5, 0.05,
        help="0 = pure collaborative filtering (what similar users liked). "
             "1 = pure content-based (genre similarity only)."
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "- *Content-based*: compares genres using TF-IDF + cosine similarity\n"
        "- *Collaborative*: compares learned rating-pattern factors (SVD) across ~6,000 users\n"
        "- *Hybrid*: weighted blend of both scores"
    )

query = st.text_input("Search for a movie you like", placeholder="e.g. Toy Story, Star Wars, Matrix")

if query:
    suggestions = rec.search_titles(query)
    if not suggestions:
        st.warning("No titles matched. Try a different search.")
    else:
        selected = st.selectbox("Select the exact title", suggestions)

        if st.button("Get Recommendations", type="primary"):
            matched_title, results = rec.recommend(selected, top_n=top_n, content_weight=content_weight)

            if results is None:
                st.error(results)
            else:
                st.success(f"Because you liked **{matched_title}**:")
                for i, row in results.iterrows():
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{i+1}. {row['title']}**")
                            st.caption(row["genres"].replace("|", " · "))
                        with col2:
                            if not pd.isna(row["avg_rating"]):
                                st.metric("Avg rating", f"{row['avg_rating']:.1f}⭐", label_visibility="collapsed")
                            st.caption(f"{int(row['rating_count']) if not pd.isna(row['rating_count']) else 0} ratings")
else:
    st.info("Start typing a movie title above to get recommendations.")
