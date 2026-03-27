
import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage



def load_pdfs(folder: str) -> list:
    """Load all PDFs from a folder."""
    path = Path(folder).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")
    
    print(f"Loading PDFs from {path}...")
    loader = PyPDFDirectoryLoader(str(path))
    docs = loader.load()
    
    if not docs:
        raise ValueError("No PDF files found in folder")
    
    print(f"Loaded {len(docs)} pages from PDFs")
    return docs


def create_vector_store(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings (this may take a moment)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created!")
    
    return vector_store


def query_rag(vector_store, query: str, top_k: int = 3):
    """Query the RAG system and return answer with sources."""
    # Retrieve similar documents
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    
    # Format context from retrieved documents
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    
    # Generate answer using Ollama
    llm = ChatOllama(model="qwen3:0.6b")
    
    prompt = f"""Use only the provided context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # Show sources
    print("\n" + "="*60)
    print("Answer:")
    print("="*60)
    print(answer)
    
    print("\n" + "="*60)
    print("Sources:")
    print("="*60)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        print(f"[{i}] {source} (Page {page})")
    
    return answer


def interactive_chat(vector_store):
    print("\n" + "="*60)
    print("RAG Chat - Type 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("Your question:(type exit or q to quit) ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!...")
                break
            if not question:
                continue
            
            query_rag(vector_store, question)
            print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Simple RAG system for chatting with your PDF files"
    )
    parser.add_argument(
        "folder",
        help="Folder containing PDF files"
    )
    parser.add_argument(
        "--question",
        help="Ask a single question and exit (optional)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of text chunks (default: 1000)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve (default: 3)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and process PDFs
        docs = load_pdfs(args.folder)
        vector_store = create_vector_store(docs, chunk_size=args.chunk_size)
        
        # Chat or single question
        if args.question:
            query_rag(vector_store, args.question, top_k=args.top_k)
        else:
            interactive_chat(vector_store)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



## HOW TO USE

# Install dependencies:

# pip install langchain langchain-community pypdf faiss-cpu ollama

# Start Ollama with the required models:

# ollama pull qwen3:0.5b
# ollama pull nomic-embed-text

# Create a folder with your PDF files:

# mkdir my_pdfs

# example : python rag_simple.py my_pdfs/


# ### Ask a single question:
# python rag_simple.py my_pdfs/ --question "What is the main topic?"

# ### Retrieve more documents:

# python rag_simple.py my_pdfs/ --top-k 5 # means 5 results for each query will be retrieved

# ## example

# $ python rag_simple.py research_papers/
# Type 'exit' to quit

