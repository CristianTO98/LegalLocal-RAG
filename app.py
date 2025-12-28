"""
This is the Chainlit UI for LegalLocal RAG.
It handles the chat interface, file uploads, and displays references.
The core RAG logic is imported from engine.py.
"""

import chainlit as cl
import engine
import asyncio
import queue
import threading

# --- Chainlit App Handlers ---

@cl.on_chat_start
async def start():
    # Show a friendly loading message
    msg = cl.Message(content="⚙️ Warming up the local AI engine...")
    await msg.send()
    
    try:
        # Load models using our engine
        embeddings = await cl.make_async(engine.load_embeddings)()
        llm = await cl.make_async(engine.load_llm)()
        cl.user_session.set("embeddings", embeddings)
        cl.user_session.set("llm", llm)
    except Exception as e:
        await cl.Message(content=f"❌ Oops, something went wrong while loading: {str(e)}").send()
        return

    msg.content = "✅ Ready! Let's analyze some documents."
    await msg.update()

    # Wait for the user to upload their PDF
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF document so I can start learning from it.",
            accept=["application/pdf"],
            max_size_mb=20
        ).send()

    pdf_file = files[0]
    
    msg = cl.Message(content=f"📚 Reading `{pdf_file.name}` and building the index...")
    await msg.send()

    # Start the RAG pipeline
    import chromadb
    
    tmp_path = pdf_file.path

    # Use engine functions for processing
    docs = await cl.make_async(engine.extract_text_from_pdf_blocks)(tmp_path)
    
    if not docs:
        msg.content = "❌ I couldn't find any text in that PDF. Is it a scanned image?"
        await msg.update()
        return
    
    parents = await cl.make_async(engine.create_parent_chunks)(docs, chunk_size=engine.CHUNKING_CONFIG["parent_chunk_size"])
    children = await cl.make_async(engine.create_child_chunks)(
        parents, 
        child_size=engine.CHUNKING_CONFIG["child_chunk_size"],
        overlap=engine.CHUNKING_CONFIG["child_chunk_overlap"]
    )
    
    # Track which pages we actually processed
    unique_pages = set()
    for c in children:
        unique_pages.update(c.get("pages", [c["page"]]))
    
    # We use a unique name for each document's collection in ChromaDB
    client = chromadb.Client()
    collection_name = f"legal_{abs(hash(pdf_file.name)) % 10000000}"
    
    # Cleanup old indexes if they exist
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    child_texts = [c["text"] for c in children]
    embeddings_model = cl.user_session.get("embeddings")
    child_embeddings = await cl.make_async(embeddings_model.encode)(child_texts, normalize_embeddings=True)
    
    collection.add(
        ids=[f"c_{i}" for i in range(len(children))],
        embeddings=child_embeddings.tolist(),
        documents=child_texts,
        metadatas=[{
            "parent_text": c["parent_text"][:4000], 
            "page": c["page"],
            "pages": ",".join(str(p) for p in c.get("pages", [c["page"]]))
        } for c in children]
    )
    
    cl.user_session.set("collection", collection)
    cl.user_session.set("pdf_path", tmp_path)
    
    # Give the user a quick summary of what we just indexed
    msg.content = f"✅ `{pdf_file.name}` is ready!\n\n📊 **Stats:** {len(unique_pages)} pages processed → {len(parents)} sections → {len(children)} searchable chunks"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    collection = cl.user_session.get("collection")
    llm = cl.user_session.get("llm")
    embeddings_model = cl.user_session.get("embeddings")

    if not collection:
        await cl.Message(content="Hey! Please upload a document first so I can help you.").send()
        return

    # Semantic search step
    query = message.content
    query_prefixed = engine.BGE_CONFIG["query_prefix"] + query
    query_emb = await cl.make_async(embeddings_model.encode)([query_prefixed], normalize_embeddings=True)
    
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=6, 
        include=["documents", "metadatas", "distances"]
    )
    
    # Let's pick the best matches and deduplicate them
    seen_parents = set()
    context_parts = []
    sources = []
    
    print(f"\n[DEBUG] Distances found: {results['distances'][0][:6]}")
    
    for i, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        
        # We ignore matches that are too far off semantically
        if distance > 0.45:
            continue
            
        p_text = metadata["parent_text"]
        if p_text not in seen_parents and len(sources) < 2:
            seen_parents.add(p_text)
            source_id = len(sources) + 1
            
            pages_str = metadata.get("pages", str(metadata["page"]))
            pages = [int(p) for p in pages_str.split(",") if p]
            page_display = f"Pages {pages[0]+1}-{pages[-1]+1}" if len(pages) > 1 else f"Page {pages[0]+1}"
            
            sources.append({
                "id": source_id,
                "text": p_text, 
                "page": metadata["page"],
                "pages": pages,
                "page_display": page_display,
                "distance": distance
            })
            context_parts.append(f"[Source {source_id}] ({page_display}):\n{p_text}")

    context = "\n\n---\n\n".join(context_parts)
    
    # Console logs for debugging the retrieval
    context_tokens = engine.count_tokens(context)
    print(f"\n[DEBUG] User Question: {query[:100]}...")
    print(f"[DEBUG] Total context size: {context_tokens} tokens")
    print(f"[DEBUG] Relevant sources: {len(sources)}")
    
    # If we couldn't find anything relevant, we let the user know
    if not sources:
        response_msg = cl.Message(content="⚠️ Sorry, I couldn't find anything in the document that directly answers that. Try rephrasing?")
        await response_msg.send()
        return
    
    # Build the prompt for the LLM
    full_prompt = f"""<|im_start|>system
{engine.LEGAL_SYSTEM_PROMPT}
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

    # We run the AI generation in a separate thread so it doesn't freeze the UI
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
            token_queue.put(None) 
            
        except Exception as e:
                token_queue.put(None)
                print(f"LLM error: {e}")
        
    thread = threading.Thread(target=generate_in_thread)
    thread.start()
    
    # Read tokens from the queue and stream them to the user
    full_answer = ""
    while True:
        try:
            token = token_queue.get(timeout=0.1)
            if token is None:
                break
            full_answer += token
            await response_msg.stream_token(token)
        except queue.Empty:
            await asyncio.sleep(0.05) 
    
    thread.join()
    answer = full_answer.strip()
    
    # Add clickable references at the end
    elements = []
    source_refs = []
    
    for s in sources:
        ref_name = f"Source {s['id']}"
        source_refs.append(f"[{ref_name}]")
        
        elements.append(
            cl.Text(
                name=ref_name,
                content=f"📄 **{s['page_display']}**\n\n---\n\n{s['text']}",
                display="side"
            )
        )
    
    if sources:
        sources_footer = "\n\n---\n📚 **References:** " + " ".join(source_refs)
        answer = answer + sources_footer
    
    # Final message update with the side panel elements
    response_msg.content = answer
    response_msg.elements = elements
    await response_msg.update()

if __name__ == "__main__":
    # Note: Chainlit is usually run via 'chainlit run app.py'
    pass
