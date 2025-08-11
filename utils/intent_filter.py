import joblib
import os
import logging
from utils.config import load_config

logger = logging.getLogger(__name__)

def load_classifier():
    """Load pre-trained classifier and vectorizer for intent classification."""
    config = load_config()
    classifier_path = config["classifier_path"]
    vectorizer_path = config["vectorizer_path"]
    
    if not os.path.exists(classifier_path) or not os.path.exists(vectorizer_path):
        logger.warning("Classifier files not found. Intent filtering will be disabled.")
        return None, None
    
    try:
        clf = joblib.load(classifier_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Intent classifier loaded successfully")
        return clf, vectorizer
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        return None, None

# Initialize classifier at module level
clf, vectorizer = load_classifier()

def is_dmrc_query(query: str) -> bool:
    """Determine if a query is related to DMRC."""
    if clf is None or vectorizer is None:
        logger.warning("Classifier not available, defaulting to True")
        return True
    
    try:
        # Clean and validate input
        query = query.strip()
        if not query:
            return False
        
        # Vectorize and predict
        vec = vectorizer.transform([query])
        prediction = clf.predict(vec)[0]
        
        logger.debug(f"Query: '{query[:50]}...' classified as: {prediction}")
        return prediction == "dmrc"
        
    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        return True  
