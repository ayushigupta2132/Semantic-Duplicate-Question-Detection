import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Semantic Question Matching",
    page_icon="🤖"
)

st.title("Semantic Question Matching")

st.write(
    """
    Enter two questions and the model will predict
    whether they are semantically duplicate.

    Model:
    SentenceTransformer (all-MiniLM-L6-v2) + XGBoost
    """
)


# Load models
embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict(q1, q2):

    q1_emb = embedder.encode([q1])
    q2_emb = embedder.encode([q2])

    abs_diff = np.abs(q1_emb - q2_emb)

    cos_sim = cosine_similarity(
        q1_emb,
        q2_emb
    )

    X = np.hstack([
        q1_emb,
        q2_emb,
        abs_diff,
        cos_sim
    ])

    prob = model.predict_proba(X)[0][1]

    return prob


q1 = st.text_area(
    "Question 1",
    height=100
)

q2 = st.text_area(
    "Question 2",
    height=100
)


if st.button("Check Duplicate"):

    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions.")

    else:

        score = predict(q1, q2)

        st.subheader("Prediction")
    
    if score >= 0.5:

        st.success("Duplicate Questions")
        st.metric(
            "Duplicate Probability",
            f"{score * 100:.2f}%"
    )
    else:
        st.error("Not Duplicate")
        st.metric(
            "Duplicate Probability",
            f"{score * 100:.2f}%"
    )
    st.progress(float(score))
        
st.markdown("---")

st.caption(
    "Trained on over 400,000 labeled question pairs."
)