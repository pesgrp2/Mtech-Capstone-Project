import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -----------------------------
# Load Data
# -----------------------------
meta_df = pd.read_excel("Sample Meta.xlsx")
reviews_df = pd.read_excel("Sample Reviews.xlsx")

# -----------------------------
# Embedding Model
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

review_texts = reviews_df["text"].fillna("").tolist()
review_embeddings = embedder.encode(review_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(review_embeddings.shape[1])
index.add(review_embeddings)

# -----------------------------
# Summarizer (LLM)
# -----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📝 Amazon Product Review Summarizer (RAG-based)")

# Dropdown for product titles
product_options = meta_df["title"].dropna().unique().tolist()
selected_product = st.selectbox("Select a Product Title:", product_options)

# Submit button
if st.button("Generate Summary"):
    # Step 1: Find product metadata
    product_row = meta_df[meta_df["title"] == selected_product]
    
    if not product_row.empty:
        asin = product_row.iloc[0]["parent_asin"]
        
        st.subheader("📦 Product Details")
        st.write(f"**Title:** {product_row.iloc[0]['title']}")
        st.write(f"**Average Rating:** {product_row.iloc[0]['average_rating']}")
        st.write(f"**Rating Count:** {product_row.iloc[0]['rating_number']}")
        st.write(f"**Features:** {product_row.iloc[0]['features']}")
        
        # Display product images if available
        if isinstance(product_row.iloc[0]["images"], list) and len(product_row.iloc[0]["images"]) > 0:
            st.subheader("🖼 Product Images")
            for img in product_row.iloc[0]["images"]:
                st.image(img["large"], width=200)
        
        # Step 2: Get reviews for this product
        product_reviews = reviews_df[reviews_df["parent_asin"] == asin]["text"].fillna("").tolist()
        
        if product_reviews:
            # Step 3: Retrieve top reviews using FAISS
            query_embedding = embedder.encode([selected_product], convert_to_numpy=True)
            D, I = index.search(query_embedding, k=5)
            top_reviews = [review_texts[i] for i in I[0] if i < len(review_texts)]
            
            # Step 4: Summarize reviews
            joined_reviews = " ".join(top_reviews)
            summary = summarizer(joined_reviews, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
            
            st.subheader("📝 Review Summary")
            st.write(summary)
        else:
            st.warning("No reviews found for this product.")
    else:
        st.error("Product not found in metadata.")