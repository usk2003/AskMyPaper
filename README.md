# AskMyPaper: Your Research Assistant

AskMyPaper is an AI-powered research assistant that helps users interact intelligently with academic papers. Built using Streamlit and integrated with advanced models and APIs, it allows you to upload a PDF and ask questions directly about the content. It can even fetch and analyze referenced papers automatically.

---

## 🔍 What is AskMyPaper?

**AskMyPaper bridges the gap between dense research papers and user questions.**  
Instead of reading an entire paper, users can upload it and ask specific questions. The app reads the document, understands the content, and provides human-like responses.

### Core Features:
- Upload a **research paper PDF**
- Ask **natural language questions**
- Receive **contextual answers**
- Automatically **extracts and downloads cited references** (DOI or arXiv)

---

## ⚙️ Core Components

AskMyPaper is built on five key technical components:

1. **Text Extraction & Chunking** – Converts the PDF into readable, manageable text chunks.
2. **Embeddings (HuggingFace)** – Transforms text into high-dimensional vectors for similarity search.
3. **FAISS Vector Store** – Efficient storage and fast retrieval of similar text chunks.
4. **Google Gemini LLM** – Generates intelligent, context-aware answers.
5. **Reference Downloaders** – Automatically fetches cited papers using DOI or arXiv links.

---

## 🤖 HuggingFace Embeddings

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`

- Converts chunks of text into numerical vectors (embeddings)
- Captures semantic meaning to enable context-aware search
- Enables similarity search to find relevant information for user queries

---

## 🧠 Google Gemini LLM

**Model Used:** `gemini-2.5-flash-preview-04-17`

- A powerful large language model from Google
- Takes your question and retrieved document context to generate accurate responses
- Handles both document-specific and general research questions

---

## 🔑 Google Gemini API

- Connects to Gemini via the `GOOGLE_API_KEY`
- The application sends the user’s question and context
- Gemini returns the generated answer which is displayed in the app

---

## 📥 Reference Downloaders (DOI & arXiv)

AskMyPaper enhances its knowledge by reading cited papers:

### DOI (Digital Object Identifier)
- Queried via: `https://doi.org/`
- Adds `Accept: application/pdf` header to retrieve the paper
- May redirect through services like Crossref or Unpaywall

### arXiv
- Constructs direct URL:  

- Downloads open-access papers directly

---

## 🔁 Workflow: From PDF to Answer

```bash
[User Uploads PDF] 
      ↓
[Text Extracted using PyPDF2]
      ↓
[References Detected via Regex]
      ↓
[Reference PDFs Downloaded (DOI/arXiv)]
      ↓
[Text Chunked using RecursiveCharacterTextSplitter]
      ↓
[Embeddings Generated via HuggingFace]
      ↓
[Stored in FAISS Vector Store]
      ↓
[User Asks Question]
      ↓
[Similar Chunks Retrieved]
      ↓
[Question + Context Sent to Gemini API]
      ↓
[Answer Generated & Displayed in Streamlit]
```

## 📌 Key Takeaways

- **Contextual Understanding**: Embeddings + vector search ensure highly relevant retrieval  
- **Intelligent Responses**: Gemini synthesizes information to provide detailed answers  
- **Expanded Research**: Automatically includes references from cited works  
- **Easy to Use**: Streamlit interface makes everything accessible with minimal setup  

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **PDF Handling**: PyPDF2  
- **Text Processing**: LangChain, RecursiveCharacterTextSplitter  
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)  
- **Vector DB**: FAISS  
- **LLM API**: Google Gemini via `google.generativeai`  
- **Reference Downloads**: `requests` with DOI & arXiv parsing  

---

## 📚 Future Improvements

- Add multi-PDF comparison  
- Enable PDF export of chat history  
- Integrate citation-based summarization  
- UI enhancements with better citation linking  

---

## 🧑‍💻 Developed By

- **22071A6651 – Sathvik Deekonda**  
- **22071A6662 – Urlana Suresh Kumar**  
- **22071A6664 – Yasaswi Kode**

_Final Year B.Tech Students – CSE (AI & ML), VNRVJIET (2022–2026)_

---

## 💬 Questions or Suggestions?

Feel free to open an **issue** or **pull request**!  
Let’s make research smarter and more interactive together. ✨
