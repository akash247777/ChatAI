import streamlit as st
import fitz  # PyMuPDF
import ollama
from PIL import Image
import pytesseract
import pdf2image
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
import io

# Set Tesseract path (update this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Downloads\\tesseract-ocr-setup-3.02.02.exe'

# Streamlit setup
st.title("Advanced ChatPDF with Gemma3:1B")
st.write("Upload a PDF and ask questions about its content")

# Session state management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# PDF processing functions
def extract_text_from_image(pdf_path):
    images = pdf2image.convert_from_bytes(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def process_pdf(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if not page_text.strip():  # If no text, try OCR
            try:
                img = page.get_pixmap()
                img_bytes = img.tobytes()
                page_text = pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes)))
            except Exception as e:
                st.warning(f"OCR failed for page {page_num+1}: {str(e)}")
        text.append((page_num+1, page_text))
    return text

# Text chunking and vector database
def build_vector_database(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([chunk[1] for chunk in text_chunks])
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, model

# Search function
def find_relevant_chunks(query, k=3):
    if not st.session_state.vector_db:
        return []
        
    query_embedding = st.session_state.embedding_model.encode([query])
    distances, indices = st.session_state.vector_db.search(query_embedding, k)
    
    relevant_chunks = []
    for idx in indices[0]:
        relevant_chunks.append(st.session_state.chunks[idx])
    return relevant_chunks

# PDF upload section
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Clear previous data
        st.session_state.document_text = ""
        st.session_state.chunks = []
        st.session_state.vector_db = None
        
        # Process PDF
        try:
            processed_text = process_pdf(uploaded_file.getvalue())
            
            # Create text chunks with page numbers
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            for page_num, page_text in processed_text:
                chunks = text_splitter.create_documents([page_text])
                for chunk in chunks:
                    st.session_state.chunks.append((page_num, chunk.page_content))
            
            # Build vector database
            if st.session_state.chunks:
                st.session_state.vector_db, st.session_state.embedding_model = build_vector_database(st.session_state.chunks)
                st.success(f"PDF processed successfully! {len(st.session_state.chunks)} chunks created")
            else:
                st.warning("No text could be extracted from the PDF")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# Chat interface
st.header("Chat with Document")
user_input = st.text_input("Ask a question", key="user_input")

if st.button("Send"):
    if not uploaded_file:
        st.warning("Please upload a PDF first")
    elif not user_input:
        st.warning("Please enter a question")
    else:
        with st.spinner("Searching document..."):
            relevant_chunks = find_relevant_chunks(user_input)
            
            # Create context with citations
            context = ""
            cited_pages = set()
            for page_num, chunk_text in relevant_chunks:
                context += f"[Page {page_num}]: {chunk_text}\n"
                cited_pages.add(page_num)
            
            # Generate prompt
            prompt = f"""
            Document Context:
            {context}
            
            Question: {user_input}
            
            Answer: Please provide a concise answer based on the document context.
            Include references to pages in parentheses, e.g. (Page 3).
            """
            
            # Generate response
            try:
                response = ollama.generate(
                    model="gemma3:1b",
                    prompt=prompt,
                    options={"temperature": 0.7, "max_tokens": 500}
                )
                
                # Add to chat history with citations
                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": response["response"],
                    "references": sorted(cited_pages)
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Display chat history
st.header("Conversation History")
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"**Q:** {chat['question']}")
    st.markdown(f"**A:** {chat['answer']}")
    st.caption(f"References: Pages {', '.join(map(str, chat['references']))}")
    st.write("---")

# Reset button
st.button("Clear Conversation", on_click=lambda: st.session_state.clear())