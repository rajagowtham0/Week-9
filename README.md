# Week-9


# Day-1
## Retrieval Engine
The retrieval module is designed as a modular pipeline for efficient similarity search and insight generation.
### Structure
1. initialize_engine() → Loads data and builds FAISS index
2. preprocess_text() → Cleans input text
3. retrieve_similar_cases() → Performs similarity search
4. generate_case_insight() → Generates structured insights
5. analyze_case() → End-to-end pipeline
### Key factors
1. Modular and reusable function design
2. Clear separation of retrieval and insight logic
3. FAISS-based fast similarity search
4. Clean and maintainable code structure

# Day-2 
## Embedding Generation
Embeddings are generated using a Sentence Transformer model.

- Model: all-MiniLM-L6-v2
- Input: Combined text (symptoms + doctor notes)
- Preprocessing: Lowercasing, whitespace normalization, and stop word handling.
- Output: Fixed-length dense vector

### Steps:
1. Combine symptoms and doctor notes
2. Normalize input text
3. Generate embedding using the model
4. Cache embedding for reuse
5. Store embedding in MongoDB
### Pipeline
Generate embedding for the input case.
1. Normalize input
2. Check cache
3. Check MongoDB
4. Generate embedding if needed
5. Store in DB + cache

### Storage:
Each case document stores:
- embedding vector
- embedding version