import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai
import hashlib
import re
from langchain.schema import Document # Ensure Document is imported
from dotenv import load_dotenv
import requests
from io import BytesIO

# --- Configuration & Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Helper Functions ---
def parse_pdf(pdf_file_object):
    """Parses a PDF file object and returns its text content."""
    reader = PdfReader(pdf_file_object)
    full_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            full_text += content + "\n"
    return full_text

def get_answer_from_gemini(context, question):
    """
    Asks Gemini to answer a question based on provided context.
    The prompt is tailored for factual extraction and summarization from research papers.
    """
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    prompt = f"""You are a highly analytical research assistant. Your task is to extract or synthesize information directly from the provided 'Context' to answer the user's 'Question'.

    Focus strictly on the factual content within the context. Do not invent information, speculate, or bring in outside knowledge. If the context does not contain enough information to fully answer the question, state that clearly and concisely.

    If the question asks for a comparison or details across multiple sources, synthesize the information from the relevant parts of the context.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Extract DOIs and arXiv IDs
def extract_references(text):
    # Match valid DOI pattern (must end with alphanum)
    doi_pattern = r"\b10\.\d{4,9}/[^\s.]+\b"
    
    # Match valid arXiv IDs like arXiv:2406.12345 or 10.48550/arXiv.2406.12345
    arxiv_pattern_1 = r"\barxiv:(\d{4}\.\d{4,5})\b"
    arxiv_pattern_2 = r"10\.48550/arXiv\.(\d{4}\.\d{4,5})\b"

    dois = re.findall(doi_pattern, text, re.IGNORECASE)
    arxiv_ids_1 = re.findall(arxiv_pattern_1, text, re.IGNORECASE)
    arxiv_ids_2 = re.findall(arxiv_pattern_2, text, re.IGNORECASE)

    # Normalize arXiv IDs to arXiv:xxxx.xxxxx format
    arxiv_ids = [f"arxiv:{aid}" for aid in set(arxiv_ids_1 + arxiv_ids_2)]

    return list(set(dois + arxiv_ids))


# Try to download PDF from arXiv or DOI (Unpaywall/CrossRef)
def download_pdf_from_reference(ref):
    try:
        if "arxiv" in ref.lower():
            # Clean and extract arXiv ID from either format
            match = re.search(r"(\d{4}\.\d{4,5})", ref)
            if not match:
                # print(f"DEBUG: No arXiv ID found in {ref}") # Debug print
                return None
            arxiv_id = match.group(1)
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            # print(f"DEBUG: Downloading from arXiv: {url}")
            response = requests.get(url)
            # print(f"DEBUG: arXiv Response Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}") 
            if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("application/pdf"):
                return BytesIO(response.content)
            # else:
                # print(f"DEBUG: arXiv download failed for {ref}: Status {response.status_code}, Type {response.headers.get('Content-Type')}")
        elif ref.startswith("10."):
            url = f"https://doi.org/{ref}"
            headers = {"Accept": "application/pdf"}
            # print(f"DEBUG: Attempting DOI download from: {url}")
            response = requests.get(url, headers=headers, allow_redirects=True)
            # print(f"DEBUG: DOI Response Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
            if "application/pdf" in response.headers.get("Content-Type", ""):
                return BytesIO(response.content)
            # else:
                # print(f"DEBUG: DOI {ref} did not return a direct PDF. Content-Type: {response.headers.get('Content-Type')}. URL: {response.url}")
        return None
    except Exception as e:
        # print(f"DEBUG: Download error for {ref}: {e}")
        return None


# --- Main Streamlit Application ---
def main():
    # Set page config without the 'icon' argument for broader compatibility
    st.set_page_config(layout="centered", page_title="Research Workbench Q&A")

    # --- Sidebar ---
    with st.sidebar:
        st.title('🔬 Research Workbench')
        st.markdown('''
        Upload your research papers to quickly find answers, extract key details,
        and compare information across your document collection.
        ''')
        add_vertical_space(2)
        st.write("Powered by LangChain, FAISS, and Google Gemini Pro")
        add_vertical_space(1)
        st.info("💡 *Tip:* Upload multiple papers for comprehensive cross-document insights!")


    # --- Main Content Area ---
    st.title("📚 Research Paper Q&A System")
    st.markdown("Upload your PDFs and ask specific questions to get direct answers from your research papers.")

    # File Uploader
    uploaded_pdfs = st.file_uploader(
        "Upload your research paper PDFs here",
        type='pdf',
        accept_multiple_files=True,
        help="You can upload one or multiple PDF research papers."
    )

    vectorstore = None # Initialize vectorstore outside the if block

    if uploaded_pdfs:
        # Generate a unique store name based on file names for caching
        # Include a version hash for the processing logic if it changes
        all_pdf_names = sorted([pdf.name for pdf in uploaded_pdfs])
        processing_logic_version = "v1.1" # Increment this if processing logic changes significantly
        store_hash_input = "".join(all_pdf_names) + processing_logic_version
        store_hash = hashlib.md5(store_hash_input.encode()).hexdigest()
        store_name = f"research_vs_{store_hash}"

        # Load or create vector store
        if "vectorstore" not in st.session_state or st.session_state.get("store_name") != store_name:
            st.session_state.downloaded_citations = {} # Reset downloaded citations on new uploads

            with st.spinner("Processing papers and building knowledge base... This might take a moment if downloading citations..."):
                all_chunks_with_metadata = []
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                
                # 1. Process uploaded PDFs
                for pdf_idx, uploaded_pdf in enumerate(uploaded_pdfs):
                    # Seek to start in case of multiple reads or if already read
                    uploaded_pdf.seek(0) 
                    text = parse_pdf(uploaded_pdf)
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        all_chunks_with_metadata.append(
                            Document(page_content=chunk, metadata={"source": uploaded_pdf.name})
                        )
                    
                    # Extract references from EACH uploaded PDF
                    references = extract_references(text)
                    for ref in references:
                        if ref not in st.session_state.downloaded_citations: # Avoid re-downloading
                            pdf_file_obj = download_pdf_from_reference(ref)
                            if pdf_file_obj:
                                st.session_state.downloaded_citations[ref] = pdf_file_obj
                                # st.markdown(f"<small>Downloaded citation: {ref}</small>", unsafe_allow_html=True) # Optional visual feedback

                # 2. Process downloaded citation PDFs
                if st.session_state.downloaded_citations:
                    st.markdown("<small>Processing downloaded citations...</small>", unsafe_allow_html=True)
                    for ref, pdf_obj in st.session_state.downloaded_citations.items():
                        pdf_obj.seek(0) # Reset pointer for parsing
                        text = parse_pdf(pdf_obj)
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            all_chunks_with_metadata.append(
                                Document(page_content=chunk, metadata={"source": ref})
                            )

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Check if cached vectorstore exists
                pickle_path = f"{store_name}.pkl"
                if os.path.exists(pickle_path):
                    with open(pickle_path, "rb") as f:
                        vectorstore = pickle.load(f)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.store_name = store_name
                else:
                    if all_chunks_with_metadata: # Only build if we have chunks
                        vectorstore = FAISS.from_documents(all_chunks_with_metadata, embedding=embeddings)
                        with open(pickle_path, "wb") as f:
                            pickle.dump(vectorstore, f)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.store_name = store_name
                    else:
                        st.warning("No text extracted from PDFs to build knowledge base.")
                        return # Exit if no documents processed

            st.info(f"✅ Ready! Knowledge base built from *{len(uploaded_pdfs)}* uploaded papers and *{len(st.session_state.downloaded_citations)}* cited papers.")
        else:
            vectorstore = st.session_state.vectorstore
            st.info(f"✅ Knowledge base for *{len(uploaded_pdfs)}* uploaded papers and *{len(st.session_state.downloaded_citations)}* cited papers loaded.")

        st.markdown("---")

        # Query Input Section
        st.subheader("❓ Ask Your Research Question")
        query = st.text_area(
            "Type your question here (e.g., 'What are the main findings of the paper?', 'How is X measured?', 'Compare methodology A and B')",
            height=80,
            placeholder="Enter your question about the uploaded papers..."
        )

        query_button = st.button("Get Answer", type="primary", use_container_width=True)

        if query_button and query:
            if vectorstore:
                with st.spinner("Searching papers and synthesizing answer..."):
                    # Use similarity_search_with_score to get scores
                    # It returns (Document, score) tuples
                    retrieved_results = vectorstore.similarity_search_with_score(query, k=8)
                    
                    # Unpack for context and display
                    context_for_llm = "\n\n".join([doc.page_content for doc, score in retrieved_results])

                    # Get answer from Gemini
                    if context_for_llm:
                        answer = get_answer_from_gemini(context_for_llm, query)
                    else:
                        answer = "I'm sorry, I couldn't find any relevant information in the uploaded papers to answer your question. The context was empty."


                    st.markdown("### ✨ AI-Generated Answer")
                    st.write(answer)

                    # Display source context snippets more compactly
                    st.markdown("---")
                    st.subheader("🔎 Supporting Snippets from Documents")
                    if retrieved_results:
                        for i, (doc, score) in enumerate(retrieved_results): # Unpack here
                            # Display score and source
                            with st.expander(f"Snippet {i+1} from: **{doc.metadata.get('source', 'Unknown')}** (Relevance: {score:.2f})"):
                                st.code(doc.page_content, language="text")
                    else:
                        st.info("No relevant snippets found for your question.")

            else:
                st.warning("Please upload PDFs first to build the knowledge base.")
        elif query_button and not query:
            st.warning("Please enter a question to get an answer.")
    else:
        st.info("Upload research papers above to start analyzing their content.")

    st.markdown("---")

    # Display downloaded citations as downloadable/viewable (separate from main workflow)
    if "downloaded_citations" in st.session_state and st.session_state.downloaded_citations:
        with st.expander("📂 Downloaded Citation Papers (for reference)"):
            if st.session_state.downloaded_citations:
                for ref, file_obj in st.session_state.downloaded_citations.items():
                    st.markdown(f"*Reference:* {ref}")
                    # Ensure the BytesIO object's pointer is at the beginning before download
                    file_obj.seek(0) 
                    st.download_button(label=f"Download {ref}",
                                      data=file_obj.getvalue(),
                                      file_name=f"{ref.replace('/', '_').replace(':', '-')}.pdf",
                                      mime="application/pdf")
            else:
                st.info("No citation papers were downloaded yet.")


    st.markdown("---")
    st.caption("This system aims to provide answers directly from your uploaded research papers and their extracted citations. It does not provide real-time information or engage in general conversation outside the document context.")

if _name_ == "_main_":
    main()
