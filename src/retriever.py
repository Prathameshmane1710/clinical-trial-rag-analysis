import chromadb
from sentence_transformers import SentenceTransformer


# Loading once at startup = instant searches after that
print("Loading BiomedBERT for retrieval...")
model = SentenceTransformer(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)
print("Model ready!")

# connect to ChromaDB once at startup not on every search request
client = chromadb.PersistentClient(path="embeddings/chroma_db")
collection = client.get_collection("clinical_trials")


def retrieve_trials(patient_query, n_results=5, filters=None):
    
    # Step 1 -: Embed the patient query
    # Convert their plain English into a 768-dim vector
    # using the same BiomedBERT model we used for storage
    query_embedding = model.encode(patient_query).tolist()
    
    # Step 2 -: Build the search parameters
    search_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        # documents = the original text we embedded
        # metadatas = nct_id, title, location etc.
        # distances = cosine distance scores
        "include": ["documents", "metadatas", "distances"]
    }
    
    # Step 3 -: Apply metadata filters if provided
    # This is the power of storing metadata alongside vectors
    # We can combine semantic search with hard filters
    # Example: find semantically similar trials but
    # only in United States AND only Phase 2 or 3
    if filters:
        search_params["where"] = filters
    
    # Step 4 -: Execute the search
    results = collection.query(**search_params)
    
    # Step 5 -: Format results into clean list of dicts
    # Right now results looks like:
    # {
    #   "metadatas": [[{...}, {...}, {...}]],  ← nested list
    #   "distances": [[0.034, 0.036, 0.038]],  ← nested list
    #   "documents": [["text1", "text2"...]]    ← nested list
    # }
    # The double nesting is because ChromaDB supports
    # multiple queries at once sending only one query
    # so we take [0] to get our results
    
    formatted_results = []
    SIMILARITY_THRESHOLD = 0.75 
    
    for i in range(len(results["metadatas"][0])):
        
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        
        # Convert distance to similarity score
        # Distance = 1 - similarity
        # So similarity = 1 - distance
        # Multiply by 100 for percentage
        similarity_score = round((1 - distance) * 100, 2)
        
        if similarity_score < SIMILARITY_THRESHOLD * 100:
            continue
        
        formatted_results.append({
            "nct_id": metadata["nct_id"],
            "title": metadata["title"],
            "status": metadata["status"],
            "condition": metadata["condition"],
            "phase": metadata["phase"],
            "location": metadata["location"],
            "last_updated": metadata["last_updated"],
            "summary": metadata["summary"],
            "eligibility": metadata["eligibility"],
            "similarity_score": similarity_score
        })
    
    return formatted_results


if __name__ == "__main__":
    
    # Test the retriever directly
    query = "45 year old male with type 2 diabetes, HbA1c 8.2, metformin stopped working"
    
    print(f"\nPatient query: {query}")
    print("\nSearching for matching trials...")
    
    trials = retrieve_trials(query, n_results=5)
    
    print(f"\nTop {len(trials)} matching trials:\n")
    for i, trial in enumerate(trials):
        print(f"Match {i+1}: {trial['title']}")
        print(f"  Similarity: {trial['similarity_score']}%")
        print(f"  Location: {trial['location']}")
        print(f"  Phase: {trial['phase']}")
        print()