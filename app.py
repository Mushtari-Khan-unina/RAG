import streamlit as st
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# Load models and tokenizers
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

# Build FAISS index
def build_faiss_index(documents):
    embeddings = [tokenizer.encode(doc, return_tensors='pt')[0].detach().numpy() for doc in documents]
    dimension = embeddings[0].shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.vstack(embeddings))
    return index

# Retrieve documents
def retrieve_documents(query, index, documents, k=5):
    query_embedding = tokenizer.encode(query, return_tensors='pt')[0].detach().numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Summarize text
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=200)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Summarize with RAG
def summarize_with_rag(query, text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    index = build_faiss_index(chunks)
    relevant_docs = retrieve_documents(query, index, chunks)
    combined_text = " ".join(relevant_docs)
    summary = summarize_text(combined_text)
    return summary

# Main function to run the app
def main():
    st.title("PDF Summarizer with RAG")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text Preview:", pdf_text[:500])  # Show part of the extracted text

        if st.button("Summarize"):
            query = "Summarize this document"
            summary = summarize_with_rag(query, pdf_text)
            st.write("Summary:", summary)

if __name__ == "__main__":
    main()
