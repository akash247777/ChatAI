# 🌟 AI Assistant & PDF Chatbot with Gemma3:1B 📚🤖

A powerful AI chat application built with **Streamlit**, **LangChain**, and **Ollama**, powered by the **Gemma3:1B** model. Engage in real-time conversations or ask questions about PDF documents with embedded OCR and semantic search capabilities.

---

### 📌 Features
- **AI Chatbot**: Conversational AI with Gemma3:1B for natural, context-aware dialogue.  
- **PDF Q&A**: Upload PDFs, extract text (even from images), and ask questions with page references.  
- **OCR Support**: Handle scanned PDFs or low-quality text using Tesseract OCR.  
- **Vector Database**: Semantic search with FAISS for relevant document chunk retrieval.  
- **Session Management**: Persistent chat history and context across sessions.  

---

### 🚀 Tech Stack
- **Frontend**: Streamlit 🌟  
- **LLM**: Gemma3:1B via Ollama 🦾  
- **OCR**: Tesseract + PyMuPDF 📄  
- **Vector DB**: FAISS + Sentence Transformers 🧠  

---

### 🌈 Demo

https://github.com/user-attachments/assets/61ff50fd-9483-4740-a30e-c05531909125

---

### 🛠️ Installation
1. **Clone the repo**:  
     ```bash
     git clone https://github.com/your-repo/ai-pdf-chat.git
     cd ai-pdf-chat
     ```

2. **Install dependencies** :
     ```bash
     pip install -r requirements.txt
     ```
3. **Setup Ollama & Gemma3** :
     ```bash
     # Install Ollama (if not done)
     curl -s https://ollama.ai/install.sh | bash
    
     # Download Gemma3:1B model
     ollama pull gemma3:1b  
    ```
4. **Configure Tesseract (Windows example)** :
  Update `pytesseract.pytesseract.tesseract_cmd` path in `app.py` to your Tesseract installation.
  Example:
  ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

🏃 Usage
1. Run the app :
```bash
streamlit run app.py
```
2. Chat with AI :
    * Go to `AI Assistant built with Gemma3:1b` tab.
    * Start a conversation!
3. PDF Q&A :
    * Go to `Advanced ChatPDF with Gemma3:1B` tab.
    * Upload a PDF → Ask questions → Get answers with page references.

# 📝 Key Code Highlights
**AI Chat Logic (in `app.py`): **     

```python
# Initialize Gemma3:1B
llm_engine = ChatOllama(
    model="gemma3:1b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# Process user input
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})
```
**PDF Processing (in `app.py`):**

```python
# Extract text with OCR fallback
def process_pdf(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if not page_text.strip():
            img = page.get_pixmap()
            page_text = pytesseract.image_to_string(Image.open(io.BytesIO(img.tobytes())))
        text.append((page_num+1, page_text))
    return text
```
