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
Generate an embedding for the input case.
1. Normalize input
2. Check cache
3. Check MongoDB
4. Generate embedding if needed
5. Store in DB + cache

### Storage:
Each case document stores:
- embedding vector
- embedding version

# Day-3
## Similar case retrieval
To retrieve the most relevant past cases for a given input. 
The system returns the top 5 similar cases along with their similarity scores in a consistent format.

### Pipeline
1. Input (Symptoms + Doctor Notes)
2. Text Preprocessing
3. Embedding Generation
4. FAISS Vector Search (Top 5)
5. Similarity Score Calculation
6. Filtering (Threshold ≥ 0.5)
7. Sorting (Descending Similarity)
8. Insight Generation
9. Structured Output

### Process
1. The input (symptoms + doctor notes) is first cleaned and normalized
2. An embedding is generated for the input text
3. FAISS is used to find the top 5 similar cases
3. A similarity score is computed from the distance
4. Low-confidence results are filtered using a threshold (≥ 0.5)
5. Duplicate case IDs are removed.
6. The final results are sorted and structured.

### Output structure
{
  "similar_cases": [
    {"case_id": "...", "similarity_score": ...},
    {"case_id": "...", "similarity_score": ...}
  ],
  "symptoms": "The similarity is mainly due to shared symptoms such as ...",
  "treatment": "In similar past cases, patients well responded ...",
  "similarity_score": "Based on the 5 similar patients, the weighted confidence score obtained is ..."
}
