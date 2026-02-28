#!/usr/bin/env python
# coding: utf-8

# In[25]:


# app.py
import streamlit as st
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



# In[ ]:


# pip install transformers sentence-transformers faiss-cpu streamlit


# In[27]:


# ----------------------------- 
# Load product and review data 
# ----------------------------- 
products_df = pd.read_csv("Appliances_meta.csv") 
reviews_df = pd.read_csv("Appliances_reviews.csv")


# In[5]:


# -----------------------------
# Build embeddings for reviews
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
review_texts = reviews_df["text"].fillna("").tolist()
review_embeddings = embedder.encode(review_texts, convert_to_numpy=True)


# In[6]:


# Create FAISS index
dimension = review_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(review_embeddings)


# In[7]:


# -----------------------------
# Summarizer model
# -----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# In[8]:


def rag_summarize(product_asin, top_k=5):
    # Get reviews for the product
    product_reviews = reviews_df[reviews_df["parent_asin"] == product_asin]
    if product_reviews.empty:
        return "No reviews available for this product."
    
    # Embed product title for retrieval
    product_title = products_df[products_df["parent_asin"] == product_asin]["title"].values[0]
    query_embedding = embedder.encode([product_title], convert_to_numpy=True)
    
    # Retrieve top reviews
    distances, indices = index.search(query_embedding, top_k)
    retrieved_reviews = [review_texts[i] for i in indices[0] if i < len(review_texts)]
    
    # Concatenate reviews
    context = " ".join(retrieved_reviews)
    
    # Summarize
    summary = summarizer(context, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    return summary


# In[9]:


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛒 Amazon Review Summarizer (RAG-powered)")


# In[10]:


# Product selection
product_options = products_df["title"].tolist()
selected_product = st.selectbox("Select a product:", product_options)


# In[ ]:


if selected_product:
    asin = products_df[products_df["title"] == selected_product]["parent_asin"].values[0]
    #st.subheader("📌 Product Details")
    #st.write(products_df[products_df["title"] == selected_product].to_dict(orient="records")[0])
    
    # Button to generate summary
    if st.button("Generate Review Summary"):
        st.subheader("📝 Review Summary")
        summary = rag_summarize(asin, top_k=5)
        st.write(summary)


# In[ ]:




