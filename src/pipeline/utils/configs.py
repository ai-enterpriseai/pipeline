import os
import torch

from typing import List, Optional, Any, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, ValidationInfo, ConfigDict, field_validator

from .logging import setup_logger

logger = setup_logger(__name__)

class ProcessorConfig(BaseModel):
    """Configuration for document processing with validation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic processing
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=100)
    max_threads: int = Field(default=4, ge=1)
    allowed_extensions: List[str] = Field(default_factory=lambda: ["pdf", "txt", "md", "doc", "docx"])
    extract_metadata: bool = Field(default=True)
    generate_chunk_context: bool = Field(default=False)
    preprocessors: List[Any] = Field(default_factory=list)  # Changed from DocumentPreprocessor for testing
    batch_size: int = Field(default=10, ge=1)
    
    # Rate limiting
    rate_limit: int = Field(default=1000, gt=0)
    
    # Timeouts
    operation_timeout: float = Field(default=3000.0, gt=0)
    
    # Chunk deduplication  
    enable_deduplication: bool = Field(default=True)
    similarity_threshold: float = Field(default=0.95, ge=0, le=1)
    
    # Backpressure
    max_queue_size: int = Field(default=1000, gt=0)
    
    # Document validation
    validation_hooks: List[Any] = Field(default_factory=list)  # Changed from DocumentHook for testing

    @field_validator("chunk_size")
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100 or v > 2000:
            raise ValueError("chunk_size must be between 100 and 2000")
        return v

    @field_validator("chunk_overlap")
    def validate_overlap(cls, v: int, info: ValidationInfo) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("max_threads")
    def validate_threads(cls, v: int) -> int:
        import os
        return min(v, os.cpu_count() or 1)

    @field_validator("rate_limit")
    def validate_rate_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rate_limit must be positive")
        return v

class EmbedderType(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"

class EmbedderConfig(BaseModel):
    """Configuration for embeddings."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Dense embedding configuration
    embedder_type: EmbedderType = Field(
        default=EmbedderType.SENTENCE_TRANSFORMER,
        description="Type of embedder to use"
    )
    dense_model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Name of the sentence-transformers model to use"
    )
    dense_model_dimension: int = Field(
        default=768,
        description="Dimension of the dense model embeddings"
    )
    device: str = Field(
        default="cpu",
        description="Device to use for dense embeddings (cpu/cuda)"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to normalize dense embeddings"
    )
    
    # Sparse embedding configuration
    sparse_model_path: Union[str, Path] = Field(
        default=Path("bm25_params.json"),
        description="Path to save/load BM25 parameters"
    )
    bm25_b: float = Field(
        default=0.75,
        description="BM25 b parameter"
    )
    bm25_k1: float = Field(
        default=1.5,
        description="BM25 k1 parameter"
    )

    # Performance settings
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    max_length: int = Field(
        default=512,
        description="Maximum sequence length for dense embeddings"
    )
    
    # Retry settings
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    min_retry_wait: float = Field(
        default=1.0,
        description="Minimum wait time between retries in seconds"
    )
    max_retry_wait: float = Field(
        default=10.0,
        description="Maximum wait time between retries in seconds"
    )

    @field_validator('sparse_model_path')
    def convert_to_path(cls, v: Union[str, Path]) -> Path:
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator('device')
    def validate_device(cls, v: str) -> str:
        if v not in ['cpu', 'cuda']:
            if torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        if v == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            return 'cpu'
        return v

class MaintenanceConfig(BaseModel):
    """Configuration for index maintenance."""
    
    enable_optimization: bool = Field(
        default=True,
        description="Enable periodic index optimization"
    )
    optimization_interval: int = Field(
        default=24,
        description="Hours between optimization runs"
    )

class IndexerConfig(BaseModel):
    """Configuration for vector indexing."""
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    # Database configuration
    use_local_db: bool = Field(
        default=False,
        description="Whether to use a local database or a cloud instance"
    )
    collection_name: str = Field(
        ...,  # Required field
        description="Name of the vector collection"
    )
    url: str = Field(
        ...,  # Required field
        description="URL of the cloud instance"
    )
    qdrant_api_key: str
    db_path: Union[str, Path] = Field(
        default=Path("database"),
        description="Path to local database"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=100,
        description="Batch size for vector uploads"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    parallel_uploads: int = Field(
        default=4,
        description="Number of parallel upload threads"
    )
    
    # Vector dimensions
    dense_dim: int = Field(
        default=768,  # Default for all-mpnet-base-v2
        description="Dimension of dense vectors"
    )
    
    # Operation timeouts
    operation_timeout: float = Field(
        default=30.0,
        description="Timeout for database operations in seconds"
    )
    
    # Maintenance configuration
    maintenance: MaintenanceConfig = Field(
        default_factory=MaintenanceConfig,
        description="Maintenance settings"
    )
    
    @field_validator('parallel_uploads')
    def validate_parallel_uploads(cls, v: int) -> int:
        max_threads = os.cpu_count() or 1
        if v < 1:
            return 1
        return min(v, max_threads)

class CacheConfig(BaseModel):
    """Configuration for query cache."""
    max_size: int = Field(
        default=1000,
        description="Maximum number of cached queries"
    )
    ttl: int = Field(
        default=3600,
        description="Time to live in seconds"
    )

class RerankerType(Enum):
    RERANKER = "cross_encoder"
    COHERE = "cohere" 
    
    # Allow any string value while still maintaining enum for documentation
    @classmethod
    def _missing_(cls, value: str) -> str:
        return value

    @property
    def default_model(self) -> str:
        defaults = {
            RerankerType.RERANKER: "cross-encoder/ms-marco-MiniLM-L-6-v2",
            RerankerType.COHERE: "rerank-multilingual-v3.0",
        }
        return defaults.get(str(self), "")

class RerankerConfig(BaseModel):
    """Configuration for reranking."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    reranker_type: RerankerType = Field(
        default=RerankerType.RERANKER,
        description="Type of reranker to use (cross_encoder/cohere)"
    )
    model_name: str = Field(
        default=None,
        description="Model name or path"
    )    
    top_k: int = Field(
        default=5,
        description="Number of documents to return after reranking"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for reranking"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for Cohere"
    )

    @field_validator('model_name')
    def set_default_model(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v is None:
            reranker_type = info.data.get('reranker_type', RerankerType.RERANKER)
            return reranker_type.default_model
        return v

class RetrieverConfig(BaseModel):
    """Configuration for retrieval."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Search parameters
    top_k: int = Field(
        default=25,
        description="Number of documents to retrieve"
    )
    
    # Reranking parameters
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig,
        description="Reranker configuration"
    )

    # reranker_model: str = Field(
    #     default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    #     description="Model to use for reranking"
    # )

    # rerank_top_k: int = Field(
    #     default=5,
    #     description="Number of documents to return after reranking"
    # )
    
    # Query processing
    query_max_length: int = Field(
        default=512,
        description="Maximum query length"
    )

    decompose_query: bool = Field(
        default=False,
        description="Whether to expand the query using LLM"
    )

    # Performance settings
    timeout: float = Field(
        default=10.0,
        description="Search timeout in seconds"
    )
    
    # Cache settings
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )

class PromptManagerConfig(BaseModel):
    """Configuration for the PromptManager."""
    
    version: str = Field(
        default="v0.1",
        description="Version of the prompt manager"
    )
    prompt: str = Field(
        default="standard",
        description="Name of the default prompt template to use"
    )
    description: str = Field(
        default="No Description",
        description="Description of the prompt manager's purpose"
    )
    required_fields: List[str] = Field(
        default_factory=list,
        description="Required fields that must be present in templates"
    )
    templates_dir: Union[str, Path] = Field(
        default=Path.cwd() / "prompts",
        description="Directory containing prompt template files"
    )
    encoding: str = Field(
        default="utf-8",
        description="Encoding for reading template files"
    )

class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    name: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0) 
    timeout: float = Field(default=30.0, gt=0.0)

    @field_validator('name')
    def validate_model_name(cls, v: str) -> str:
        allowed_models = {
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "claude-3-5-sonnet-20241022"
        }
        if v not in allowed_models:
            raise ValueError(f"Invalid model name. Must be one of: {allowed_models}")
        return v

class LLMConfig(BaseModel):
    """LLM configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    primary_model: ModelConfig
    fallback_model: ModelConfig
    together_api_key: str
    anthropic_api_key: str
    together_base_url: str = Field(
        default="https://api.together.ai/v1"
    )
    max_retries: int = Field(default=3, gt=0)
    response_model: Optional[BaseModel] = None # TODO change carefully, as everything fails 

    @field_validator('together_api_key', 'anthropic_api_key')
    def validate_api_keys(cls, v: str) -> str:
        if not v or len(v) < 10:  # Basic validation
            raise ValueError("Invalid API key")
        return v

class PipelineConfig(BaseModel):
    """Pipeline component configurations mapping to pipeline configs."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    processor: ProcessorConfig = Field(
        description="Document processor settings"
    )
    indexer: IndexerConfig = Field(
        description="Vector store settings"
    )
    embedder: EmbedderConfig = Field(
        description="Embedding model settings"
    )
    retriever: RetrieverConfig = Field(
        description="Retrieval settings"
    )
    manager: PromptManagerConfig = Field(
        description="Prompt manager settings"
    )
    generator: LLMConfig = Field(
        description="Generation settings"
    )
