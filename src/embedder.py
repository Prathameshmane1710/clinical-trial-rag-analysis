import json
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_model():

    print("Loading BiomedBERT model...")
    print("First time = downloads ~400MB, after that instant from cache")

    model = SentenceTransformer("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

    print("Model loaded successfully!")
    return model

def setup_chromadb(persist_path="embeddings/chroma_db"):
    print(f"Setting up ChromaDB at {persist_path}...")

    # Creating the folder if it doesn't exist
    os.makedirs(persist_path, exist_ok=True)

    # PersistentClient means ChromaDB saves to disk
    # not just in memory, survives after script ends
    client = chromadb.PersistentClient(path=persist_path)
    
    # A collection is like a table in a database
    # get_or_create means:
    # if collection exists already -: use it
    # if not -: create it fresh
    # This avoids duplicates if we run the script twice
    collection = client.get_or_create_collection(
        name="clinical_trials",
        metadata={"hnsw:space": "cosine"}
        # hnsw:space = "cosine" tells ChromaDB to use cosine similarity when searching
        # This matches how BiomedBERT vectors work
    )
    
    print("ChromaDB ready!")
    return client, collection


def chunk_trial(trial, chunk_size=400, chunk_overlap=50):
    """
    Splits trial text into multiple chunks
    each within BiomedBERT's 512 token limit
    
    chunk_size=400 words ≈ 380-400 tokens
    stays safely under 512 limit
    
    chunk_overlap=50 words overlap between chunks
    Why overlap? So context isn't lost at boundaries
    Example without overlap:
    Chunk 1 ends: "...age must be between"
    Chunk 2 starts: "18 and 65 years old..."
    → split breaks the criterion ❌
    
    With overlap:
    Chunk 1 ends: "...age must be between 18"
    Chunk 2 starts: "...must be between 18 and 65..."
    → criterion preserved in both chunks ✅
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # Split the eligibility text specifically
    # not the whole text_to_embed
    # Why? Title and condition are short
    # Only eligibility needs splitting
    eligibility = trial["eligibility"]
    chunks = splitter.split_text(eligibility)
    
    # Create one document per chunk
    # each with the title+condition prefix
    # so every chunk has full context
    chunked_trials = []
    for idx, chunk in enumerate(chunks):
        chunked_trials.append({
            **trial,  # copy all fields
            "chunk_index": idx,
            "chunk_id": f"{trial['nct_id']}_chunk_{idx}",
            "text_to_embed": f"Title: {trial['title']}. "
                           f"Condition: {trial['condition']}. "
                           f"Eligibility: {chunk}"
        })
    
    return chunked_trials

def embed_and_store(trials, collection, model, batch_size=32):
    
    print(f"Starting embedding of {len(trials)} trials...")
    print(f"Processing in batches of {batch_size}")
    
    # We process in batches not one by one
    # Loading 32 texts at once into the model
    # is much faster than 32 separate model calls
    
    total_embedded = 0
    
    for i in range(0, len(trials), batch_size):
        
        # Slice the trials list into batches
        # Batch 1: trials[0:32]
        # Batch 2: trials[32:64]
        # Batch 3: trials[64:96] etc.
        batch = trials[i: i + batch_size]
        
        # Extracting the text to embed from each trial
        texts = [trial["text_to_embed"] for trial in batch]
        
        # Extract unique IDs for each chunk
        # Format: NCT04123456_chunk_0
        # this format so we can find and delete
        # all chunks of a trial by filtering on nct_id
        ids = [trial["chunk_id"] for trial in batch]
        
        # Extract metadata for each trial
        # This gets stored alongside the vector in ChromaDB
        # We can filter search results by these fields later
        # e.g. "only find trials in United States"
        # or "only find Phase 2 trials"
        metadatas = [
            {
                "nct_id": trial["nct_id"],
                "chunk_index": trial["chunk_index"],
                "title": trial["title"],
                "status": trial["status"],
                "condition": trial["condition"],
                "phase": trial["phase"],
                "location": trial["location"],
                "last_updated": trial["last_updated"],
                "summary": trial["summary"],
                "eligibility": trial["eligibility"]
            }
            for trial in batch
        ]
        
        # model.encode() runs each text through BiomedBERT
        # and returns a 768-dimensional vector for each
        # show_progress_bar=False because we have our own logging
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
        # .tolist() converts numpy arrays to plain Python lists
        # ChromaDB needs plain Python lists not numpy arrays
        
        # Checks if any of these IDs already exist in ChromaDB
        # This handles the case if embedder.py runs twice
        # So no duplicates vectors for the same trial
        existing = collection.get(ids=ids)
        existing_ids = set(existing["ids"])
        
        # Filter out trials that are already embedded
        new_indices = [
            j for j, id_ in enumerate(ids) 
            if id_ not in existing_ids
        ]
        
        if new_indices:
            # Only add the trials that aren't already in ChromaDB
            collection.add(
                documents=[texts[j] for j in new_indices],
                embeddings=[embeddings[j] for j in new_indices],
                metadatas=[metadatas[j] for j in new_indices],
                ids=[ids[j] for j in new_indices]
            )
        
        total_embedded += len(new_indices)
        skipped = len(batch) - len(new_indices)
        
        print(
            f"Batch {i//batch_size + 1} done — "
            f"embedded: {total_embedded} | "
            f"skipped (already exists): {skipped}"
        )
    
    print(f"\nEmbedding complete!")
    print(f"Total vectors in ChromaDB: {collection.count()}")
    return total_embedded


def main():
    
    # Step 1 -: Loading parsed trials from disk
    print("Loading parsed trials...")
    with open("data/trials_parsed.json", "r", encoding="utf-8") as f:
        trials = json.load(f)
    print(f"Loaded {len(trials)} trials")

    # NEW — chunk trials before embedding
    print("Chunking trials...")
    all_chunks = []
    for trial in trials:
        chunks = chunk_trial(trial)
        all_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Average chunks per trial: {len(all_chunks)/len(trials):.1f}")
    
    # Step 2 -: Load BiomedBERT
    model = load_model()
    
    # Step 3 -: Setup ChromaDB
    client, collection = setup_chromadb()
    
    # Step 4 -: Embed and store everything
    embed_and_store(all_chunks, collection, model)
    
    print("vector database is ready at embeddings/chroma_db/")

if __name__ == "__main__":
    main()
