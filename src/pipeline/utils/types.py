import uuid
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field, UUID4, ConfigDict

T = TypeVar('T')

class ChunkMetadata(BaseModel):
    """Metadata for a single document chunk."""
    id: UUID4 = Field(default_factory=uuid.uuid4)
    start: int = 0
    end: int = 0
    page: int = 1
    source: str = "unknown"
    section: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)

class DocumentMetadata(BaseModel):
    """Metadata for a complete document."""
    id: UUID4 = Field(default_factory=uuid.uuid4)
    title: str = ""
    author: str = ""
    date: str = ""
    source: str = ""
    pages: int = 0
    language: str = ""
    file_type: str = ""
    creation_date: str = ""
    modification_date: str = ""
    size_bytes: int = 0
    processed_at: datetime = Field(default_factory=datetime.now)

class ProcessedDocument(BaseModel):
    """Processed document with chunks and metadata."""
    chunks: List[Tuple[str, ChunkMetadata]]
    metadata: DocumentMetadata
    source: str

class CacheEntry(BaseModel, Generic[T]):
    """Cache entry with expiration."""
    value: T
    expiry: datetime
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expiry
    
class IndexStats(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    total_uploads: int = 0
    last_update: datetime = None

    def update_uploads(self, n_uploads: int):
        self.total_uploads += n_uploads
        self.last_update = datetime.now()

    def update_documents(self, n_documents: int, n_chunks: int):
        self.total_documents += n_documents
        self.total_chunks += n_chunks
        self.last_update = datetime.now()

class VectorMetadata(BaseModel):
    """Metadata associated with a vector."""
    document_id: UUID4 = Field(default_factory=uuid.uuid4)
    chunk_metadata: ChunkMetadata 
    source: str
    embedding_timestamp: datetime
    chunk_text: str

class SearchResult(BaseModel):
    """Container for search results."""
    text: str
    score: float
    metadata: Dict[str, Any]
    original_rank: int
    reranked_score: Optional[float] = None

class DecomposedQuery(BaseModel):
    """Structured output for query expansion."""
    original_query: str = Field(
        ...,
        description="The original input query"
    )
    sub_queries: List[str] = Field(
        ...,
        description="List of expanded sub-queries",
        min_items=1,
        max_items=5
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how the query was broken down"
    )

class LLMResponse(BaseModel):
    """Structured LLM response."""
    content: str
    model: str
    tokens: int
    citations: List[str] = Field(default_factory=list)
    finish_reason: Optional[str] = None

class ChatMessage(BaseModel):
    """Enhanced chat message with proper structure."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, query: str, context: Optional[str] = None) -> "ChatMessage":
        """Create a user message with optional context."""
        if context:
            enhanced_content = f"{context}\n\nUser Query:\n{query}" # TODO enclose 
        else:
            enhanced_content = query
        return cls(role="user", content=enhanced_content)
    
    @classmethod
    def assistant(cls, content: str, metadata: Optional[Dict[str, Any]] = None) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content, metadata=metadata or {})


class QueryGenerationResult(BaseModel):
    """Result from query generation."""
    query: str
    description: str = ""
    expected_info: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
