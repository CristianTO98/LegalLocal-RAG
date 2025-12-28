"""
This is the core engine of LegalLocal RAG. 
It handles document processing, model loading, and semantic search.
Everything here is designed to work offline.
"""

import os
import warnings
from pathlib import Path
from typing import List, Dict, Any
import tiktoken

# Let's clean up the console by hiding some of the noisier warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Basic folder setup for models and cache
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"

MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Tokenizer setup (standard cl100k_base used by Qwen and GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Returns the number of tokens in a string."""
    return len(tokenizer.encode(text))

# --- Configuration ---

# Here we define which model we're using and how to load it
MODEL_CONFIG = {
    "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "name": "Qwen 3 4B",
    "context_size": 4096,
}

# Settings for the BGE embedding model
BGE_CONFIG = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "query_prefix": "Represent this sentence for searching relevant passages: ",
}

# How we want to split the documents (Measured in TOKENS now)
CHUNKING_CONFIG = {
    "parent_chunk_size": 550,   # 550 tokens ≈ 2000-2200 caracteres
    "child_chunk_size": 150,    # 150 tokens para la búsqueda
    "child_chunk_overlap": 50,  # 50 tokens de solapamiento
}

# The personality and rules for our AI assistant
LEGAL_SYSTEM_PROMPT = """You are a specialized Legal Research Assistant. Provide accurate responses derived EXCLUSIVELY from the provided context.

STRICT RULES:
1. ONLY use provided context. If not found, say "Information not found in the provided documents."
2. NEVER fabricate information.
3. ALWAYS cite specific sections or pages using format: [Source X] where X is the source number.
4. Maintain absolute objectivity.

RESPONSE FORMAT:
- Direct answer.
- Supporting evidence.
- Citations in format: [Source 1], [Source 2], etc."""

# --- Document Processing Logic ---

def extract_text_from_pdf_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Grabs text from the PDF using PyMuPDF. 
    It groups text by page to make sure we don't lose the context of where things are.
    """
    import fitz
    documents = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            page_texts = []
            
            for block in blocks:
                if len(block) >= 5:
                    x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                    if len(block) >= 7 and block[6] != 0: continue
                    text = str(text).strip()
                    if len(text) < 10: continue
                    is_header_footer = (y0 < page.rect.height * 0.05 or y1 > page.rect.height * 0.95)
                    if is_header_footer and len(text) < 30: continue
                    page_texts.append(text)
            
            if page_texts:
                combined = "\n\n".join(page_texts)
                documents.append({"text": combined, "page": page_num})
        
        doc.close()
        print(f"[PDF Extract] Total pages: {total_pages}, Pages with text: {len(documents)}")
        
    except Exception as e:
        print(f"Error extracting PDF: {str(e)}")
        return []
    
    return documents

def create_parent_chunks(documents: List[Dict[str, Any]], chunk_size: int = 550) -> List[Dict[str, Any]]:
    """
    Groups pages together into 'Parent' chunks for better AI context.
    Uses token count instead of characters for precision.
    """
    if not documents: return []
    
    parent_chunks = []
    current_chunk = ""
    current_pages = []
    
    for doc in documents:
        page_text = doc["text"]
        page_num = doc["page"]
        
        # We estimate tokens by adding the new page text
        if current_chunk and count_tokens(current_chunk) + count_tokens(page_text) > chunk_size:
            parent_chunks.append({
                "text": current_chunk.strip(),
                "page": current_pages[0],
                "pages": list(current_pages)
            })
            current_chunk = ""
            current_pages = []
        
        current_chunk += page_text + "\n\n"
        current_pages.append(page_num)
    
    if current_chunk.strip():
        parent_chunks.append({
            "text": current_chunk.strip(),
            "page": current_pages[0],
            "pages": list(current_pages)
        })
    
    unique_pages = set()
    for chunk in parent_chunks: unique_pages.update(chunk.get("pages", [chunk["page"]]))
    print(f"[Chunking] Parent chunks: {len(parent_chunks)}, Pages covered: {len(unique_pages)}")
    return parent_chunks

def create_child_chunks(parent_chunks: List[Dict[str, Any]], child_size: int = 150, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Splits Parent chunks into smaller 'Child' chunks for precise searching.
    Uses token-based splitting for consistency.
    """
    child_chunks = []
    for p_idx, parent in enumerate(parent_chunks):
        p_text = parent["text"]
        p_page = parent["page"]
        p_pages = parent.get("pages", [p_page])
        
        # Get tokens for the whole parent
        tokens = tokenizer.encode(p_text)
        
        if len(tokens) <= child_size:
            child_chunks.append({"text": p_text, "parent_idx": p_idx, "parent_text": p_text, "page": p_page, "pages": p_pages})
            continue
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + child_size, len(tokens))
            
            # Decode the slice of tokens back to text
            child_text = tokenizer.decode(tokens[start_idx:end_idx]).strip()
            
            if child_text:
                child_chunks.append({"text": child_text, "parent_idx": p_idx, "parent_text": p_text, "page": p_page, "pages": p_pages})
            
            start_idx = end_idx - overlap
            if end_idx >= len(tokens): break
            
    print(f"[Chunking] Child chunks: {len(child_chunks)}")
    return child_chunks

# --- Model Loading ---

def load_embeddings():
    """Load the BGE embeddings model on the CPU."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_CONFIG["model_name"], cache_folder=str(CACHE_DIR / "embeddings"))
    return model.cpu()

def load_llm():
    """Initialize the local Qwen model using llama-cpp-python."""
    from llama_cpp import Llama
    model_path = MODELS_DIR / MODEL_CONFIG["filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
    return Llama(
        model_path=str(model_path.absolute()),
        n_ctx=MODEL_CONFIG["context_size"],
        n_threads=4,
        n_batch=2048,
        verbose=False
    )

