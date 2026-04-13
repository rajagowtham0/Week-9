# vector_index.py

# IMPORTS
import numpy as np
import faiss
import logging

# VECTOR INDEX CLASS
class VectorIndex: # handles FAISS-based vector indexing and similarity search

    def __init__(self):
        self.index = None      # FAISS index
        self.case_ids = []     # Mapping of index → case_id
    # BUILD FAISS INDEX
    def build_index(self, embeddings, case_ids): # build FAISS index using stored embeddings

        # Parameters list of embedding vectors and corresponding case identifiers

        # Validate input
        if not embeddings or not case_ids:
            raise ValueError("Embeddings or case_ids are empty")

        if len(embeddings) != len(case_ids):
            raise ValueError("Embeddings and case_ids length mismatch")

        # Convert to numpy float32 (required for FAISS)
        embeddings = np.array(embeddings).astype("float32")

        # Get embedding dimension
        dimension = embeddings.shape[1]

        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)

        # Add vectors to index
        self.index.add(embeddings)

        # Store case_id mapping
        self.case_ids = case_ids

        logging.info(f"FAISS index built with {self.index.ntotal} vectors")

    # SEARCH SIMILAR CASES
    def search(self, query_embedding, top_k=5): # perform similarity search using FAISS
    
        # Parameters embedding of the input query and number of similar cases to retrieve (default = 5)
        # returns a list of dictionaries with case IDs and similarity score.
       
        # Ensure index is initialized
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")

        # Reshape if single vector
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Convert to float32
        query_embedding = query_embedding.astype("float32")

        # Perform FAISS search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        # Iterate through retrieved indices
        for rank, idx in enumerate(indices[0]):

            # Safety check (FAISS may return -1)
            if idx == -1:
                continue

            case_id = self.case_ids[idx]

            # Convert distance to similarity score
            # Smaller distance is equal to higher similarity
            score = 1 / (1 + distances[0][rank])

            results.append({
                "case_id": case_id,
                "similarity_score": round(float(score), 4)
            })

        return results