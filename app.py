"""
================================================================================
LegalLocal RAG v2.0 - Chainlit Edition
================================================================================

SYSTEM OVERVIEW:
This application provides a 100% offline RAG (Retrieval-Augmented Generation)
system for legal professionals using Chainlit for a modern chat interface.

ARCHITECTURE:
- UI: Chainlit (Modern chat with side-by-side citations)
- Embeddings: BAAI/bge-small-en-v1.5 (Sentence-Transformers)
- Vector Store: ChromaDB (Local/In-memory)
- LLM: Qwen 3 4B (via llama-cpp-python)
- Strategy: Parent-Child Chunking (Precise search -> Full context)
================================================================================
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import chainlit as cl

# ==============================================================================
# WARNING SUPPRESSION
# ==============================================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"

MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "name": "Qwen 3 4B",
    "context_size": 4096,
}

BGE_CONFIG = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "query_prefix": "Represent this sentence for searching relevant passages: ",
}

CHUNKING_CONFIG = {
    "parent_chunk_size": 1000,  # Smaller parents for better relevance
    "child_chunk_size": 300,    # Smaller children for precise search
    "child_chunk_overlap": 50,
}

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

# ==============================================================================
# CORE RAG LOGIC (Preserved from original)
# ==============================================================================

def extract_text_from_pdf_blocks(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF blocks.
    Returns one document per PAGE (not per block) to preserve page integrity.
    """
    import fitz
    documents = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            page_height = page.rect.height
            page_texts = []
            
            for block in blocks:
                if len(block) >= 5:
                    x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                    
                    # Skip images (block_type != 0)
                    if len(block) >= 7 and block[6] != 0:
                        continue
                    
                    text = str(text).strip()
                    
                    # Skip very short blocks (page numbers, etc)
                    if len(text) < 10:
                        continue
                    
                    # Only skip header/footer if VERY short
                    is_header_footer = (y0 < page_height * 0.05 or y1 > page_height * 0.95)
                    if is_header_footer and len(text) < 30:
                        continue
                    
                    page_texts.append(text)
            
            # Combine all text from the page into ONE document
            if page_texts:
                combined = "\n\n".join(page_texts)
                documents.append({"text": combined, "page": page_num})
        
        doc.close()
        
        # Log extraction stats
        pages_extracted = len(documents)
        print(f"[PDF Extract] Total pages: {total_pages}, Pages with text: {pages_extracted}")
        
    except Exception as e:
        print(f"Error extracting PDF: {str(e)}")
        return []
    
    return documents

def create_parent_chunks(documents: List[Dict[str, Any]], chunk_size: int = 1536) -> List[Dict[str, Any]]:
    """
    Create parent chunks preserving page information.
    Each document entry now represents ONE page, so page mapping is straightforward.
    """
    if not documents:
        return []
    
    parent_chunks = []
    current_chunk = ""
    current_pages = []
    
    for doc in documents:
        page_text = doc["text"]
        page_num = doc["page"]
        
        # If adding this page would exceed chunk_size, finalize current chunk
        if current_chunk and len(current_chunk) + len(page_text) > chunk_size:
            parent_chunks.append({
                "text": current_chunk.strip(),
                "page": current_pages[0] if current_pages else 0,
                "pages": list(current_pages)  # All pages in this chunk
            })
            current_chunk = ""
            current_pages = []
        
        current_chunk += page_text + "\n\n"
        current_pages.append(page_num)
    
    # Don't forget the last chunk
    if current_chunk.strip():
        parent_chunks.append({
            "text": current_chunk.strip(),
            "page": current_pages[0] if current_pages else 0,
            "pages": list(current_pages)
        })
    
    # Log chunking stats
    unique_pages = set()
    for chunk in parent_chunks:
        unique_pages.update(chunk.get("pages", [chunk["page"]]))
    
    print(f"[Chunking] Parent chunks: {len(parent_chunks)}, Pages covered: {len(unique_pages)}")
    
    return parent_chunks

def create_child_chunks(parent_chunks: List[Dict[str, Any]], child_size: int = 384, overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Create child chunks from parent chunks, preserving page information.
    """
    child_chunks = []
    
    for p_idx, parent in enumerate(parent_chunks):
        p_text = parent["text"]
        p_page = parent["page"]
        p_pages = parent.get("pages", [p_page])
        
        if len(p_text) <= child_size:
            child_chunks.append({
                "text": p_text, 
                "parent_idx": p_idx, 
                "parent_text": p_text, 
                "page": p_page,
                "pages": p_pages
            })
            continue
        
        start = 0
        while start < len(p_text):
            end = min(start + child_size, len(p_text))
            
            # Try to break at sentence boundary
            if end < len(p_text):
                last_period = p_text[start:end].rfind(". ")
                if last_period > child_size // 2:
                    end = start + last_period + 2
            
            child_text = p_text[start:end].strip()
            if child_text:
                child_chunks.append({
                    "text": child_text, 
                    "parent_idx": p_idx, 
                    "parent_text": p_text, 
                    "page": p_page,
                    "pages": p_pages
                })
            
            start = end - overlap
            if end >= len(p_text):
                break
    
    print(f"[Chunking] Child chunks: {len(child_chunks)}")
    return child_chunks

# ==============================================================================
# MODEL LOADERS
# ==============================================================================

def load_embeddings():
    """Load BGE embeddings model."""
    from sentence_transformers import SentenceTransformer
    
    # Simple load - let sentence-transformers handle everything
    model = SentenceTransformer(
        BGE_CONFIG["model_name"], 
        cache_folder=str(CACHE_DIR / "embeddings")
    )
    # Explicitly move to CPU
    model = model.cpu()
    return model

def load_llm():
    from llama_cpp import Llama
    model_path = MODELS_DIR / MODEL_CONFIG["filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return Llama(
        model_path=str(model_path.absolute()),
        n_ctx=MODEL_CONFIG["context_size"],
        n_threads=4,
        n_batch=256,
        verbose=False
    )

# ==============================================================================
# CHAINLIT HANDLERS
# ==============================================================================

@cl.on_chat_start
async def start():
    # Load Models
    msg = cl.Message(content="⚙️ Initializing LegalLocal RAG Engine...")
    await msg.send()
    
    try:
        embeddings = await cl.make_async(load_embeddings)()
        llm = await cl.make_async(load_llm)()
        cl.user_session.set("embeddings", embeddings)
        cl.user_session.set("llm", llm)
    except Exception as e:
        await cl.Message(content=f"❌ Error loading models: {str(e)}").send()
        return

    msg.content = "✅ Engine Ready."
    await msg.update()

    # Ask for PDF
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a legal PDF document to begin analysis.",
            accept=["application/pdf"],
            max_size_mb=20
        ).send()

    pdf_file = files[0]
    
    msg = cl.Message(content=f"📚 Processing `{pdf_file.name}` with Parent-Child indexing...")
    await msg.send()

    # Process Document
    import chromadb
    
    tmp_path = pdf_file.path

    docs = await cl.make_async(extract_text_from_pdf_blocks)(tmp_path)
    
    if not docs:
        msg.content = "❌ Could not extract text from PDF. Is it a scanned document?"
        await msg.update()
        return
    
    parents = await cl.make_async(create_parent_chunks)(docs)
    children = await cl.make_async(create_child_chunks)(parents)
    
    # Calculate stats
    unique_pages = set()
    for c in children:
        unique_pages.update(c.get("pages", [c["page"]]))
    
    # Create Index
    client = chromadb.Client()
    collection_name = f"legal_{abs(hash(pdf_file.name)) % 10000000}"
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    
    child_texts = [c["text"] for c in children]
    # Embed children
    embeddings_model = cl.user_session.get("embeddings")
    child_embeddings = await cl.make_async(embeddings_model.encode)(child_texts, normalize_embeddings=True)
    
    collection.add(
        ids=[f"c_{i}" for i in range(len(children))],
        embeddings=child_embeddings.tolist(),
        documents=child_texts,
        metadatas=[{
            "parent_text": c["parent_text"][:4000],  # Limit to avoid ChromaDB issues
            "page": c["page"],
            "pages": ",".join(str(p) for p in c.get("pages", [c["page"]]))
        } for c in children]
    )
    
    cl.user_session.set("collection", collection)
    cl.user_session.set("pdf_path", tmp_path)
    
    # Show stats to user
    msg.content = f"✅ `{pdf_file.name}` indexed!\n\n📊 **Stats:** {len(unique_pages)} pages processed → {len(parents)} parent chunks → {len(children)} searchable segments"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    collection = cl.user_session.get("collection")
    llm = cl.user_session.get("llm")
    embeddings_model = cl.user_session.get("embeddings")

    if not collection:
        await cl.Message(content="Please upload a document first.").send()
        return

    # Search
    query = message.content
    query_prefixed = BGE_CONFIG["query_prefix"] + query
    query_emb = await cl.make_async(embeddings_model.encode)([query_prefixed], normalize_embeddings=True)
    
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=10,  # Get more results to find better matches
        include=["documents", "metadatas", "distances"]
    )

    # Deduplicate and build context - filter by distance threshold
    seen_parents = set()
    context_parts = []
    sources = []
    
    print(f"\n[DEBUG] All distances: {results['distances'][0][:6]}")
    
    for i, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        
        # Skip if too far (cosine distance > 0.5 means not very similar)
        if distance > 0.6:
            print(f"  [SKIP] Distance {distance:.3f} too high for: {metadata['parent_text'][:50]}...")
            continue
            
        p_text = metadata["parent_text"]
        if p_text not in seen_parents and len(sources) < 4:
            seen_parents.add(p_text)
            source_id = len(sources) + 1
            
            # Parse pages from metadata
            pages_str = metadata.get("pages", str(metadata["page"]))
            pages = [int(p) for p in pages_str.split(",") if p]
            page_display = f"Pages {pages[0]+1}-{pages[-1]+1}" if len(pages) > 1 else f"Page {pages[0]+1}"
            
            sources.append({
                "id": source_id,
                "text": p_text[:2000],  # Limit each source to 2000 chars
                "page": metadata["page"],
                "pages": pages,
                "page_display": page_display,
                "distance": distance
            })
            context_parts.append(f"[Source {source_id}] ({page_display}):\n{p_text[:2000]}")




    context = "\n\n---\n\n".join(context_parts)
    
    # Debug: Log context length and preview
    print(f"\n[DEBUG] Query: {query[:100]}...")
    print(f"[DEBUG] Context length: {len(context)} chars")
    print(f"[DEBUG] Sources found: {len(sources)}")
    for s in sources:
        print(f"  - Source {s['id']} ({s['page_display']}, dist={s['distance']:.3f}): {s['text'][:80]}...")
    
    # Handle case where no relevant sources found
    if not sources:
        response_msg = cl.Message(content="⚠️ No relevant information found in the document for this query. Try rephrasing your question.")
        await response_msg.send()
        return
    
    # Generate
    full_prompt = f"""<|im_start|>system
{LEGAL_SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
CONTEXT:
{context}

QUESTION:
{query}
<|im_end|>
<|im_start|>assistant
"""
    
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Run LLM in a thread to avoid blocking the event loop
    import asyncio
    import queue
    import threading
    
    token_queue = queue.Queue()
    
    def generate_in_thread():
        try:
            for chunk in llm(
                full_prompt,
                max_tokens=1024,
                temperature=0,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
                stream=True
            ):
                token = chunk["choices"][0]["text"]
                token_queue.put(token)
            token_queue.put(None)  # Signal completion
        except Exception as e:
            token_queue.put(None)
            print(f"LLM error: {e}")
    
    # Start generation thread
    thread = threading.Thread(target=generate_in_thread)
    thread.start()
    
    # Stream tokens as they arrive
    full_answer = ""
    while True:
        try:
            token = token_queue.get(timeout=0.1)
            if token is None:
                break
            full_answer += token
            await response_msg.stream_token(token)
        except queue.Empty:
            await asyncio.sleep(0.05)  # Yield control to event loop
    
    thread.join()
    answer = full_answer.strip()

    
    # Create citations as Chainlit elements
    elements = []
    source_refs = []
    
    for s in sources:
        # Create a clickable reference name
        ref_name = f"Source {s['id']}"
        source_refs.append(f"[{ref_name}]")
        
        elements.append(
            cl.Text(
                name=ref_name,
                content=f"📄 **{s['page_display']}**\n\n---\n\n{s['text']}",
                display="side"
            )
        )
    
    # Append source references to the answer so they're always clickable
    if sources:
        sources_footer = "\n\n---\n📚 **References:** " + " ".join(source_refs)
        answer = answer + sources_footer
    
    # Final update with citations attached
    response_msg.content = answer
    response_msg.elements = elements
    await response_msg.update()

if __name__ == "__main__":
    # Note: Chainlit is usually run via 'chainlit run app.py'
    # But we can include this for direct execution if needed
    pass
