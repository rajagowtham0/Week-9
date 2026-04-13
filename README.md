# Week-9

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