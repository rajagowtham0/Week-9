# embedding.py


# IMPORTS
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from pymongo import MongoClient

from utils.config import (
    EMBEDDING_MODEL_NAME,
    MONGO_URI,
    DATABASE_NAME,
    COLLECTION_NAME
)

# MONGODB CONNECTION
# Used to store and retrieve embeddings
client = MongoClient(MONGO_URI)
collection = client[DATABASE_NAME][COLLECTION_NAME]
# IN-MEMORY CACHE
# Avoid recomputing embeddings for the same input
embedding_cache = {}
# CACHE KEY GENERATION
def generate_cache_key(text: str) -> str: # generate a unique key for input text, which ensures the same input → same cache key
    normalized = text.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()
# CACHE RETRIEVAL
def get_cached_embedding(text: str): # retrieve embeddings from in memory cache
    key = generate_cache_key(text)
    return embedding_cache.get(key)
# CACHE STORAGE
def store_embedding(text: str, embedding): # store embeddings in cache
    key = generate_cache_key(text)
    embedding_cache[key] = embedding
# MODEL LOADING (ONLY ONCE)
_model = None

def load_embedding_model(): # load sentence transformer model only once. prevents reloading on every request
    global _model

    if _model is None:
        logging.info(f"Model loaded: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model
# MAIN EMBEDDING FUNCTION
def generate_embedding(symptoms: str, doctor_notes: str) -> np.ndarray: # main embedding execution function

    # Input validation
    if not isinstance(symptoms, str) or not isinstance(doctor_notes, str):
        raise TypeError("Inputs must be strings.")

    # Normalize input for consistency
    combined_text = f"{symptoms}. {doctor_notes}".lower().strip()
    # 1. CHECK CACHE
    cached = get_cached_embedding(combined_text)
    if cached is not None:
        logging.info("Embedding fetched from cache")
        return cached

    # 2. CHECK MONGODB
    doc = collection.find_one({
        "symptoms": symptoms,
        "doctor_notes": doctor_notes
    })

    if doc and "embedding" in doc:
        logging.info("Embedding fetched from MongoDB")

        embedding = np.array(doc["embedding"], dtype=np.float32)

        # Store in cache for faster future access
        store_embedding(combined_text, embedding)

        return embedding

    # 3. GENERATE EMBEDDING
    model = load_embedding_model()

    logging.info(f"Generating embedding using: {EMBEDDING_MODEL_NAME}")

    embedding = model.encode(
        combined_text,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype(np.float32)   # Ensure FAISS compatibility (similarity search)
    # 4. STORE IN MONGODB
    collection.update_one(    # clean structured way
        {
            "symptoms": symptoms,
            "doctor_notes": doctor_notes
        },
        {
            "$set": {
                "symptoms": symptoms,
                "doctor_notes": doctor_notes,
                "embedding": embedding.tolist(),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_version": EMBEDDING_MODEL_NAME
            }
        },
        upsert=True
    )

    # 5. STORE IN CACHE
    store_embedding(combined_text, embedding)

    return embedding
# COMBINE INPUT TEXT
def combine_text(symptoms: str, doctor_notes: str) -> str: # combine symptoms and doctor notes into a single string.

    if not isinstance(symptoms, str) or not isinstance(doctor_notes, str):
        raise TypeError("Both inputs must be strings.")

    return f"{symptoms}. {doctor_notes}"