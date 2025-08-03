import time
import uuid
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationEntry:
    """Represents a single conversation entry."""
    user_query: str
    bot_response: str
    timestamp: float
    source: str
    confidence: float
    context_used: List[Tuple[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserSession:
    """Represents a user session with memory management."""
    session_id: str
    created_at: float
    last_accessed: float
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=20))
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_conversation(self, entry: ConversationEntry) -> None:
        """Add a conversation entry to the session."""
        self.conversation_history.append(entry)
        self.last_accessed = time.time()
        self._update_stats(entry)
    
    def get_recent_conversations(self, count: int = 5) -> List[ConversationEntry]:
        """Get recent conversation entries."""
        return list(self.conversation_history)[-count:]
    
    def get_conversation_context(self, count: int = 3) -> str:
        """Get formatted conversation context for LLM prompt."""
        recent = self.get_recent_conversations(count)
        if not recent:
            return ""
        
        context = "Recent conversation context:\n"
        for entry in recent:
            context += f"{entry.user_query}\n{entry.bot_response}\n"
        return context + "\n"
    
    def _update_stats(self, entry: ConversationEntry) -> None:
        """Update session statistics."""
        if "total_conversations" not in self.session_stats:
            self.session_stats["total_conversations"] = 0
        if "avg_confidence" not in self.session_stats:
            self.session_stats["avg_confidence"] = 0.0
        
        self.session_stats["total_conversations"] += 1
        # Update average confidence
        current_avg = self.session_stats["avg_confidence"]
        total = self.session_stats["total_conversations"]
        self.session_stats["avg_confidence"] = (current_avg * (total - 1) + entry.confidence) / total
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if session has expired."""
        return time.time() - self.last_accessed > ttl_seconds
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            "session_id": self.session_id,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "last_accessed": datetime.fromtimestamp(self.last_accessed).isoformat(),
            "total_conversations": len(self.conversation_history),
            "stats": self.session_stats,
            "preferences": self.user_preferences
        }

class SessionMemory:
    """
    Main session memory manager for the DMRC chatbot.
    
    Features:
    - Session isolation using unique IDs
    - Automatic cleanup of expired sessions
    - Memory size limits
    - Conversation context building
    - User preference tracking
    """
    
    def __init__(self, max_sessions: int = 100, ttl_seconds: int = 3600):
        """
        Initialize session memory manager.
        
        Args:
            max_sessions: Maximum number of active sessions
            ttl_seconds: Time-to-live for sessions in seconds
        """
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.stats = {
            "total_sessions_created": 0,
            "total_sessions_expired": 0,
            "total_conversations": 0
        }
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            str: Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, returning existing session")
            return session_id
        
        # Clean up expired sessions before creating new one
        self._cleanup_expired_sessions()
        
        # Check if we're at capacity
        if len(self.sessions) >= self.max_sessions:
            self._remove_oldest_session()
        
        # Create new session
        self.sessions[session_id] = UserSession(
            session_id=session_id,
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        self.stats["total_sessions_created"] += 1
        logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def add_conversation(self, session_id: str, user_query: str, bot_response: str, 
                        source: str = "unknown", confidence: float = 0.0,
                        context_used: Optional[List[Tuple[str, str]]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a conversation entry to a session.
        
        Args:
            session_id: Session ID
            user_query: User's query
            bot_response: Bot's response
            source: Response source (e.g., 'dmrc_rag', 'gemini_fallback')
            confidence: Confidence score
            context_used: Context used for response
            metadata: Additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new session")
            self.create_session(session_id)
        
        entry = ConversationEntry(
            user_query=user_query,
            bot_response=bot_response,
            timestamp=time.time(),
            source=source,
            confidence=confidence,
            context_used=context_used or [],
            metadata=metadata or {}
        )
        
        self.sessions[session_id].add_conversation(entry)
        self.stats["total_conversations"] += 1
        
        logger.debug(f"Added conversation to session {session_id}: {user_query[:50]}...")
        return True
    
    def get_conversation_context(self, session_id: str, count: int = 3) -> str:
        """
        Get conversation context for LLM prompt.
        
        Args:
            session_id: Session ID
            count: Number of recent conversations to include
            
        Returns:
            str: Formatted conversation context
        """
        if session_id not in self.sessions:
            return ""
        
        return self.sessions[session_id].get_conversation_context(count)
    
    def get_recent_conversations(self, session_id: str, count: int = 5) -> List[ConversationEntry]:
        """
        Get recent conversation entries for a session.
        
        Args:
            session_id: Session ID
            count: Number of conversations to return
            
        Returns:
            List[ConversationEntry]: Recent conversation entries
        """
        if session_id not in self.sessions:
            return []
        
        return self.sessions[session_id].get_recent_conversations(count)
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update user preferences for a session.
        
        Args:
            session_id: Session ID
            preferences: User preferences dictionary
            
        Returns:
            bool: True if successful
        """
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].user_preferences.update(preferences)
        self.sessions[session_id].last_accessed = time.time()
        return True
    
    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """
        Get user preferences for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict[str, Any]: User preferences
        """
        if session_id not in self.sessions:
            return {}
        
        return self.sessions[session_id].user_preferences.copy()
    
    def reset_session(self, session_id: str) -> bool:
        """
        Reset a session (clear conversation history).
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if successful
        """
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].conversation_history.clear()
        self.sessions[session_id].session_stats.clear()
        self.sessions[session_id].last_accessed = time.time()
        
        logger.info(f"Reset session: {session_id}")
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if successful
        """
        if session_id not in self.sessions:
            return False
        
        del self.sessions[session_id]
        logger.info(f"Deleted session: {session_id}")
        return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[Dict[str, Any]]: Session information
        """
        if session_id not in self.sessions:
            return None
        
        return self.sessions[session_id].get_session_info()
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sessions.
        
        Returns:
            List[Dict[str, Any]]: List of session information
        """
        return [session.get_session_info() for session in self.sessions.values()]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory manager statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        return {
            **self.stats,
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "ttl_seconds": self.ttl_seconds
        }
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session.is_expired(self.ttl_seconds)
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            self.stats["total_sessions_expired"] += 1
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _remove_oldest_session(self) -> None:
        """Remove the oldest session when at capacity."""
        if not self.sessions:
            return
        
        oldest_session_id = min(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid].last_accessed
        )
        
        del self.sessions[oldest_session_id]
        logger.info(f"Removed oldest session: {oldest_session_id}")

# Global instance for easy access
session_memory = SessionMemory() 