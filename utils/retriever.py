import joblib
import numpy as np
import os
import logging
from utils.embedder import embed_query
from utils.config import load_config
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def load_vector_store():
    """Load pre-computed embeddings and FAQ data from pickle file."""
    config = load_config()
    vector_store_path = config["vector_store_path"]
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}. Run embed_and_index.py first.")
    
    try:
        corpus, embeddings, answers = joblib.load(vector_store_path)
        logger.info(f"Loaded vector store with {len(corpus)} documents")
        return corpus, embeddings, answers
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")

# Initialize vector store at module level
try:
    corpus, embeddings, answers = load_vector_store()
except Exception as e:
    logger.error(f"Failed to initialize retriever: {e}")
    corpus, embeddings, answers = [], [], []

def retrieve_top_k(query, k=3, threshold=0.55):
    """
    Retrieve top-k most similar FAQ entries for a user query.
    
    Args:
        query (str): User query
        k (int): Number of top results to return
        threshold (float): Minimum similarity score threshold
    
    Returns:
        list: List of (question, answer) tuples
    """
    if not corpus or not embeddings:
        logger.error("Vector store not properly loaded")
        return []
    
    try:
        # Embed the query
        query_vec = np.array(embed_query(query)).reshape(1, -1)
        corpus_vecs = np.array(embeddings)
        
        # Calculate similarities
        scores = cosine_similarity(query_vec, corpus_vecs).flatten()
        
        # Get top-k indices
        top_idx = scores.argsort()[::-1][:k]
        
        # Filter by threshold
        results = []
        for idx in top_idx:
            if scores[idx] >= threshold:
                results.append((corpus[idx], answers[idx]))
        
        logger.info(f"Retrieved {len(results)} results for query: {query[:50]}...")
        return results
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return []
