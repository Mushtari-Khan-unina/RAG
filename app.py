import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

# Summarize text
def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=200)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Summarize long text by splitting it into chunks
def summarize_long_text(text, chunk_size=1000):
    # Split the text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [summarize_text(chunk) for chunk in chunks]
    combined_summary = " ".join(summaries)
    return combined_summary

# Main function to run the app
def main():
    st.title("PDF Summarizer")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text.strip():
            st.write("Extracted Text Preview:", pdf_text[:500])  # Show part of the extracted text

            if st.button("Summarize"):
                summary = summarize_long_text(pdf_text)
                st.write("Summary:", summary)
        else:
            st.error("Failed to extract text from the PDF. Please try another file.")

if __name__ == "__main__":
    main()
