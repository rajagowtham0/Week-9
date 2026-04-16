# Week-9

# CCMS_AI Application
## Project overview
The CCMS AI system is designed to retrieve similar patient cases based on input symptoms and doctor notes. 
It uses embedding-based similarity and FAISS search to identify relevant past cases and generate structured insights.
## Project structure
ccms_ai/
1. data_processing/
1. database.py            # Handles MongoDB connection and data retrieval
2. models/
1. models.py              # Defines API request, response schemas, and top similar cases retrieval
3. retrieval/
1. retrieval_engine.py    # Core pipeline: retrieval + insight generation
2. vector_index.py        # FAISS index creation and similarity search
4. utils/
1. embedding.py           # Embedding generation, caching, and storage
2. config.py              # Configuration settings (DB, model, parameters)
5. app.py              # FastAPI application

## Module overview
### data_processing/database.py
Manages database operations, including fetching stored cases and embeddings.
### models/models.py
Defines structured request and response formats for the API.
### retrieval/retrieval_engine.py
Implements the main pipeline for retrieving similar cases and generating insights.
### retrieval/vector_index.py
Handles FAISS-based indexing and similarity search.
### utils/embedding.py
Generates embeddings and manages caching and storage.
### utils/config.py
Stores configuration variables such as database connection and model details.
### app.py
Serves as the main API layer connecting all components.


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

# Day-4
## Retrieval Pipeline
To build and verify a complete pipeline that processes a case description and retrieves the most similar past cases.
The system ensures consistent retrieval using embeddings and FAISS-based similarity search.

### Pipeline
1. case description
2. embedding
3. retrieval
4. similar cases
5. insight generation

### Process
1. The case description is formed by combining symptoms and doctor notes
2. The input text is converted into an embedding using a pre-trained model
3. FAISS is used to retrieve the top 5 similar cases
4. Similarity scores are calculated for each retrieved case
5. Low-confidence results are filtered, and duplicates are removed
6. The final results are structured and returned as output

# Day-5
1. Tested the developed ccms_ai system with nearly 20 sample request schemas.
2. Used sample cases are stored in the testing_samplecase_dataset.txt
3. All the requests and response schemas are cleanly documented in the Week-9 report
