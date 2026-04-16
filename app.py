import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Load API key from .env file ──────────────────────────────────────────────
load_dotenv()

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)

# ── Custom styling ───────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stChatMessage { border-radius: 12px; }
        h1 { color: #818cf8; }
    </style>
""", unsafe_allow_html=True)

# ── Session state setup ──────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


# ── RAG prompt template ──────────────────────────────────────────────────────
# This is the instruction we give the model every time a question is asked.
# {context} gets replaced with the retrieved chunks.
# {question} gets replaced with what the user typed.
PROMPT_TEMPLATE = """
You are StudyMate AI, a helpful study assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I couldn't find that in the uploaded document. Try asking something else."

Context:
{context}

Chat history:
{chat_history}

Question: {question}

Answer:
"""


# ── Core function: process the PDF ──────────────────────────────────────────
def process_pdf(pdf_file):
    """
    Takes an uploaded PDF and builds the retriever:
    1. Save to a temp file so PyPDFLoader can read it
    2. Extract all text from every page
    3. Split text into overlapping chunks
    4. Embed chunks using OpenAI embeddings
    5. Store in ChromaDB and return a retriever
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=".chroma"
    )

    os.unlink(tmp_path)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ── Core function: ask a question ────────────────────────────────────────────
def ask_question(retriever, question, chat_history):
    """
    Runs the RAG pipeline for a single question:
    1. Retrieve the 4 most relevant chunks from ChromaDB
    2. Format the prompt with context + history + question
    3. Send to GPT-4o-mini
    4. Return the answer as a string
    """
    # Format previous chat history for context
    history_text = ""
    for q, a in chat_history[-3:]:   # Only last 3 exchanges to keep it focused
        history_text += f"User: {q}\nAssistant: {a}\n"

    # Build the chain using LangChain Expression Language (LCEL)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    parser = StrOutputParser()

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: history_text
        }
        | prompt
        | llm
        | parser
    )

    return chain.invoke(question)


# ── UI: Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 StudyMate AI")
    st.markdown("*Chat with your study material*")
    st.divider()

    st.markdown("### Upload your document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a textbook chapter, lecture notes, or any study material"
    )

    if uploaded_file:
        if (not st.session_state.pdf_processed or
                st.session_state.pdf_name != uploaded_file.name):
            with st.spinner("Reading document and building knowledge base..."):
                st.session_state.retriever = process_pdf(uploaded_file)
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
            st.success(f"Ready! Ask me anything about **{uploaded_file.name}**")
        else:
            st.success(f"Loaded: **{uploaded_file.name}**")

    st.divider()
    st.markdown("### Tips")
    st.markdown("""
    - Ask questions about the content
    - Ask for a summary of a topic
    - Say *"Quiz me on this document"*
    - Say *"Give me 5 flashcards"*
    """)

    if st.session_state.pdf_processed:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── UI: Main chat area ───────────────────────────────────────────────────────
st.markdown("## 📚 StudyMate AI")
st.markdown("Upload a study document in the sidebar, then ask questions, request summaries, or generate a quiz.")
st.divider()

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Display existing chat history
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    # Chat input
    user_question = st.chat_input("Ask a question about your document...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(
                    st.session_state.retriever,
                    user_question,
                    st.session_state.chat_history
                )
            st.write(answer)

        st.session_state.chat_history.append((user_question, answer))
