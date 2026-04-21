##This is the Test.py to test multiple components of the system, This file is not meant to be included in the final project, 
## but is a playground for testing different parts of the system during development.

import json

# with open('data/trials_raw.json', 'r', encoding='utf-8') as f:
#     trials = json.load(f)

# print(f'Total trials downloaded: {len(trials)}')
# print('---FIRST TRIAL RAW STRUCTURE---')
# print(json.dumps(trials[0], indent=2))

# add this to test_data.py temporarily


# with open('data/trials_parsed.json', 'r', encoding='utf-8') as f:
#     trials = json.load(f)

# print(f'Total parsed trials: {len(trials)}')
# print('---FIRST PARSED TRIAL---')
# print(json.dumps(trials[0], indent=2))

# add to test_data.py temporarily
import chromadb
from sentence_transformers import SentenceTransformer

# Load the SAME model we used for storing
model = SentenceTransformer(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)

client = chromadb.PersistentClient(path="embeddings/chroma_db")
collection = client.get_collection("clinical_trials")

print(f"Total vectors in DB: {collection.count()}")

# Manually embed the query using BiomedBERT
query = "adult patient with type 2 diabetes, HbA1c high, metformin not working"
query_embedding = model.encode(query).tolist()

# Now search using the embedding directly
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n--- TOP 3 MATCHING TRIALS ---")
for i, metadata in enumerate(results["metadatas"][0]):
    print(f"\nMatch {i+1}:")
    print(f"  Title: {metadata['title']}")
    print(f"  Location: {metadata['location']}")
    print(f"  Distance: {results['distances'][0][i]:.4f}")


# import chromadb

# # Initialize the client (ensure path matches your setup)
# client = chromadb.PersistentClient(path="embeddings/chroma_db")
# collection = collection = client.get_collection(name="clinical_trials")

# def verify_db_health():
#     # 1. Check total count
#     count = collection.count()
#     print(f"--- Database Stats ---")
#     print(f"Total records in ChromaDB: {count}")
    
#     # 2. Check for potential ID duplicates (Logic check)
#     # We fetch a small sample to see if the metadata/IDs look correct
#     results = collection.get(limit=10)
#     print(f"\n--- Sample IDs ---")
#     print(results['ids'])

#     # 3. Test Embedding Stability (Self-Similarity Test)
#     # We query for an existing record using its own ID. 
#     # If the distance is 0.0 or very close, the embeddings are stable.
#     if count > 0:
#         test_id = results['ids'][0]
#         sample_record = collection.get(ids=[test_id], include=['embeddings', 'documents'])
        
#         query_res = collection.query(
#             query_embeddings=sample_record['embeddings'],
#             n_results=1
#         )
        
#         distance = query_res['distances'][0][0]
#         print(f"\n--- Stability Test ---")
#         print(f"Querying ID: {test_id}")
#         print(f"Distance to itself: {distance}")
        
#         if distance < 1e-5:
#             print("✅ Status: Embeddings are stable. The distance to self is near zero.")
#         else:
#             print("⚠️ Status: Warning. High distance detected. Check your embedding function.")



# import chromadb
# from collections import defaultdict

# def find_chroma_duplicates(collection_name, db_path="embeddings/chroma_db"):
#     client = chromadb.PersistentClient(path=db_path)
#     collection = client.get_collection(name=collection_name)
    
#     # Fetch all data (IDs and Documents)
#     # Note: If your DB is huge, you may need to do this in batches
#     results = collection.get(include=['documents'])
    
#     ids = results['ids']
#     docs = results['documents']
    
#     # Map document content to a list of IDs that contain it
#     content_map = defaultdict(list)
#     for i, doc in enumerate(docs):
#         content_map[doc].append(ids[i])
    
#     # Filter for content that appears more than once
#     duplicates = {text: id_list for text, id_list in content_map.items() if len(id_list) > 1}
    
#     print(f"--- Duplicate Scan Results ---")
#     if not duplicates:
#         print("✅ No duplicate content found. Every record has unique text.")
#     else:
#         print(f"⚠️ Found {len(duplicates)} pieces of text that are repeated across different IDs.")
#         for text, id_list in list(duplicates.items())[:5]:  # Show first 5 examples
#             print(f"\nText (snippet): {text[:100]}...")
#             print(f"Found in IDs: {id_list}")
            
#     return duplicates

# # Usage
# find_chroma_duplicates("clinical_trials")
# verify_db_health()
