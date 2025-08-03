from sentence_transformers import SentenceTransformer
from utils.config import load_config

# Initialize sentence transformer model for text embeddings
config = load_config()
model = SentenceTransformer(config["embedding_model"])


def embed_query(text):
    """Convert user query to embedding vector for similarity search."""
    return model.encode("query: " + text, normalize_embeddings=True).tolist()


def embed_passages(passages):
    """Convert FAQ passages to embedding vectors for storage."""
    prefixed = ["passage: " + p for p in passages]
    return model.encode(prefixed, normalize_embeddings=True).tolist()
