# StudyMate AI

A RAG-powered study assistant that lets you chat with your documents. Upload any PDF — lecture notes, textbooks, research papers — and ask questions, request summaries, or generate quizzes instantly.

**Live Demo:** https://studymate-ai-i4wtfqeim2zch8fnnpantb.streamlit.app/

---

## What It Does

- Upload any PDF up to 200MB
- Ask questions about the content in plain English
- Request topic summaries
- Generate quiz questions and flashcards
- Maintains context across multiple questions in the same session

## How It Works

1. You upload a PDF
2. The app splits it into chunks and converts them into embeddings using OpenAI
3. Those embeddings are stored in a local ChromaDB vector database
4. When you ask a question, the 4 most relevant chunks are retrieved
5. GPT-4o-mini uses those chunks to generate a grounded answer
6. It will not hallucinate answers — if the answer isn't in the document, it says so

This pattern is called **RAG (Retrieval-Augmented Generation)**.

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | ChromaDB |
| Orchestration | LangChain (LCEL) |
| PDF Parsing | PyPDFLoader |
| Environment | python-dotenv |

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/DataBuster204/studymate-ai.git
cd studymate-ai
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your OpenAI API key**

Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your-key-here
```

**5. Run the app**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Project Structure

```
studymate-ai/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed to GitHub)
├── .gitignore          # Excludes .env, venv, ChromaDB files
└── .chroma/            # Local vector store (auto-created on first run)
```

## Built By

**Olumide Daramola** — NVIDIA-certified Generative AI Developer  
[Portfolio](https://olumidedaramola.dev) · [GitHub](https://github.com/DataBuster204)
