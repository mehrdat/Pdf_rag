# Simple RAG for PDFs

A minimal, RAG system for chatting with PDF files using LangChain and Ollama.

**Key features:**
- Load entire folders of PDFs automatically
- Chat with your documents interactively
- Show sources for each answer
- ~100 lines of clean, readable code

## Setup

1. Install dependencies:
```bash
pip install langchain langchain-community pypdf faiss-cpu ollama
```

2. Start Ollama with the required models:
```bash
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

3. Create a folder with your PDF files:
```bash
mkdir my_pdfs
# Add your PDF files to my_pdfs/
```

## Usage

### Interactive chat:
```bash
python rag_simple.py my_pdfs/
```

### Ask a single question:
```bash
python rag_simple.py my_pdfs/ --question "What is the main topic?"
```

### Adjust chunk size (how much context to consider):
```bash
python rag_simple.py my_pdfs/ --chunk-size 2000
```

### Retrieve more documents:
```bash
python rag_simple.py my_pdfs/ --top-k 5
```

## How it works

1. **Load** → Reads all PDFs from your folder
2. **Split** → Breaks documents into ~1000 character chunks
3. **Embed** → Converts chunks to vectors using `nomic-embed-text`
4. **Store** → Saves vectors in FAISS (local, fast search)
5. **Retrieve** → For each question, finds the 3 most relevant chunks
6. **Answer** → Uses `qwen3:0.5b` to generate answers from context

## Example

```bash
$ python rag_simple.py research_papers/

============================================================
RAG Chat - Type 'exit' to quit
============================================================
