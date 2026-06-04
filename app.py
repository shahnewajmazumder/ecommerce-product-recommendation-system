import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

st.set_page_config(page_title="E-Commerce Product Recommendation System", page_icon="🛒")
st.title("🛒 E-Commerce Product Recommendation System")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("ratings.csv")
    products = pd.read_csv("products.csv")
    return df, products

df, products = load_data()

# --- KPI Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", int(df["User_ID"].nunique()))
col2.metric("Total Products", int(df["Product_ID"].nunique()))
col3.metric("Total Ratings", int(len(df)))

st.subheader("Ratings Distribution")
st.bar_chart(df["Rating"].value_counts().sort_index())

# --- Train Collaborative Filtering Model (SVD) ---
@st.cache_resource
def train_svd(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["User_ID", "Product_ID", "Rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model, testset

model, testset = train_svd(df)

# --- Build Content-Based Model (TF-IDF on product features) ---
@st.cache_resource
def build_content_model(products):
    products = products.copy()
    products["features"] = (
        products["Category"] + " " +
        products["Brand"] + " " +
        products["Price_Range"] + " " +
        products["Tags"]
    )
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(products["features"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=products["Product_ID"], columns=products["Product_ID"])
    return sim_df

sim_df = build_content_model(products)

# --- Evaluation ---
def evaluate(model, testset, threshold=3.5, k=5):
    predictions = model.test(testset)
    user_est_true = defaultdict(list)
    for pred in predictions:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions, recalls, ndcgs = [], [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum(1 for _, true_r in user_ratings if true_r >= threshold)
        n_rel_and_rec_k = sum(1 for est, true_r in top_k if est >= threshold and true_r >= threshold)

        precisions.append(n_rel_and_rec_k / k if k > 0 else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel > 0 else 0)

        dcg = sum(
            (2 ** (1 if true_r >= threshold else 0) - 1) / np.log2(i + 2)
            for i, (_, true_r) in enumerate(top_k)
        )
        ideal = sum((2 ** 1 - 1) / np.log2(i + 2) for i in range(min(n_rel, k)))
        ndcgs.append(dcg / ideal if ideal > 0 else 0)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

precision, recall, ndcg = evaluate(model, testset)

st.subheader("📊 Model Evaluation (on test split)")
e1, e2, e3 = st.columns(3)
e1.metric("Precision@5", f"{precision:.2f}")
e2.metric("Recall@5", f"{recall:.2f}")
e3.metric("NDCG@5", f"{ndcg:.2f}")

# --- Recommendation UI ---
st.subheader("🔍 Get Recommendations")
user_id = st.number_input("Enter User ID", min_value=1, max_value=int(df["User_ID"].max()), step=1)

tab1, tab2 = st.tabs(["🤝 Collaborative Filtering (SVD)", "🏷️ Content-Based Filtering"])

with tab1:
    if st.button("Get Collaborative Recommendations"):
        all_products = df["Product_ID"].unique()
        rated_by_user = df[df["User_ID"] == user_id]["Product_ID"].tolist()
        unrated = [p for p in all_products if p not in rated_by_user]

        recommendations = sorted(
            [(p, round(float(model.predict(user_id, p).est), 2)) for p in unrated],
            key=lambda x: x[1], reverse=True
        )

        st.subheader("Top Collaborative Recommendations")
        if recommendations:
            for product, score in recommendations[:5]:
                st.write(f"✅ **{product}** | Predicted Rating: ⭐ {score}")
        else:
            st.info("This user has already rated all products.")

        with st.expander("Products already rated by this user"):
            st.dataframe(df[df["User_ID"] == user_id][["Product_ID", "Rating"]].reset_index(drop=True))

with tab2:
    rated_products = df[df["User_ID"] == user_id]["Product_ID"].tolist()
    if rated_products:
        selected_product = st.selectbox("Pick a product you liked to find similar ones", rated_products)
        if st.button("Get Content-Based Recommendations"):
            if selected_product in sim_df.index:
                similar = (
                    sim_df[selected_product]
                    .drop(index=selected_product)
                    .sort_values(ascending=False)
                    .head(5)
                )
                st.subheader(f"Products similar to **{selected_product}**")
                for product, score in similar.items():
                    st.write(f"✅ **{product}** | Similarity Score: {score:.2f}")
            else:
                st.warning("Product not found in content database.")
    else:
        st.info("No ratings found for this user.")
