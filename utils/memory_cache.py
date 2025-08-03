import time
from collections import OrderedDict
from typing import Dict, Any, Optional

class ConversationCache:
    """Lightweight in-memory cache for conversation context with TTL."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        self._cleanup_expired()
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with timestamp."""
        self._cleanup_expired()
        
        # Remove if key exists
        if key in self.cache:
            self.cache.pop(key)
            self.timestamps.pop(key, None)
        
        # Add new entry
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
        # Remove oldest if cache is full
        if len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key, None)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        self._cleanup_expired()
        return len(self.cache)
    
    def keys(self) -> list:
        """Get all valid keys."""
        self._cleanup_expired()
        return list(self.cache.keys())

# Global cache instance
conversation_cache = ConversationCache(max_size=50, ttl_seconds=1800)  # 30 minutes TTL

def get_conversation_context(session_id: str) -> Optional[Dict]:
    """Get conversation context for a session."""
    return conversation_cache.get(f"context_{session_id}")

def set_conversation_context(session_id: str, context: Dict):
    """Set conversation context for a session."""
    conversation_cache.set(f"context_{session_id}", context)

def clear_conversation_context(session_id: str):
    """Clear conversation context for a session."""
    conversation_cache.set(f"context_{session_id}", None) 