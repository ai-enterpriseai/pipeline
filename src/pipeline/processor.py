import time
import asyncio
import hashlib
import uuid

from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    AsyncGenerator,
    TypeVar,
    Any,
    Set,
    Callable,
    Awaitable,
    TypeAlias,
)

from pydantic import BaseModel, Field
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from async_timeout import timeout

from .utils.logging import setup_logger
from .utils.types import DocumentMetadata, ProcessedDocument, ChunkMetadata
from .utils.configs import ProcessorConfig

# Type aliases
PathLike: TypeAlias = Union[str, Path]
DocumentHook: TypeAlias = Callable[[Document], Awaitable[Document]]
ValidationResult = Tuple[bool, Optional[str]]

T = TypeVar("T", bound=Document)

logger = setup_logger(__name__)

class ProcessingMetrics(BaseModel):
    start_time: float = Field(default_factory=time.time)
    loaded_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    errors: List[str] = Field(default_factory=list)
    memory_usage: List[Tuple[float, float]] = Field(default_factory=list)
    deduplication_stats: Dict[str, int] = Field(
        default_factory=lambda: {"total_chunks": 0, "duplicate_chunks": 0}
    )

    def add_error(self, error: str) -> None:
        """Add error to metrics."""
        self.errors.append(error)
        logger.error(f"Processing error: {error}")

    def update_chunks(self, count: int, duplicates: int = 0) -> None:
        """Update chunk count and deduplication stats."""
        self.total_chunks += count
        self.deduplication_stats["total_chunks"] += count
        self.deduplication_stats["duplicate_chunks"] += duplicates

    def update_loaded_documents(self, count: int = 1) -> None:
        """Update loaded documents count."""
        self.loaded_documents += count

    def update_processed_documents(self, count: int = 1) -> None:
        """Update processed documents count."""
        self.processed_documents += count

    def record_memory(self) -> None:
        """Record current memory usage."""
        import psutil

        process = psutil.Process()
        self.memory_usage.append(
            (time.time() - self.start_time, process.memory_info().rss / 1024 / 1024)
        )

    def get_deduplication_ratio(self) -> float:
        """Calculate deduplication ratio."""
        total = self.deduplication_stats["total_chunks"]
        if total == 0:
            return 0.0
        return self.deduplication_stats["duplicate_chunks"] / total

class RateLimiter:
    """Rate limiter with configurable limits and metrics."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._last_reset = time.monotonic()
        self._calls = 0
        self._lock = asyncio.Lock()
        self._metrics = {
            "total_requests": 0,
            "throttled_requests": 0,
            "last_throttle": None
        }

    async def acquire(self) -> bool:
        """
        Attempt to acquire rate limit permit.
        
        Returns:
            bool: True if acquired, False if should retry
        """
        async with self._lock:
            now = time.monotonic()
            self._metrics["total_requests"] += 1
            
            # Reset counter if period has passed
            if now - self._last_reset >= 60:
                self._calls = 0
                self._last_reset = now
            
            # Check if limit reached
            if self._calls >= self.config.rate_limit:
                self._metrics["throttled_requests"] += 1
                self._metrics["last_throttle"] = now
                return False
            
            self._calls += 1
            return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics."""
        return {
            **self._metrics,
            "current_usage": self._calls,
            "limit": self.config.rate_limit
        }
    
class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class InvalidSourceError(ProcessingError):
    """Raised when source format is invalid."""
    pass


class ChunkingError(ProcessingError):
    """Raised when chunking fails."""
    pass


class ValidationError(ProcessingError):
    """Raised when document validation fails."""
    pass


class TimeoutError(ProcessingError):
    """Raised when operation times out."""
    pass

class Processor:
    """
    Handles all document processing operations including loading,
    chunking, and metadata extraction.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        """
        Initialize processor with configuration.

        Args:
            config: Processing configuration parameters
        """
        self.config = config or ProcessorConfig()
        self._rate_limiter = RateLimiter(self.config)
        self._text_splitter = self._initialize_splitter()
        self.metrics = ProcessingMetrics()
        self._processing_lock = asyncio.Lock()
        self._cancel_event = asyncio.Event()
        self._chunk_hashes: Set[str] = set()
        self._queue: Optional[asyncio.Queue] = None

    def _initialize_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize text splitter with configuration."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

    async def process_source(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> AsyncGenerator[ProcessedDocument, None]:
        """
        Process documents from source.

        Args:
            source: File path, directory path, or list of paths

        Yields:
            Processed documents
        """
        try:
            documents = await self.load_documents(source)
            async for processed_doc in self.process_documents(documents):
                yield processed_doc
        except Exception as e:
            self.metrics.add_error(str(e))
            raise ProcessingError(f"Failed to process source: {e}") from e

    async def load_documents(
        self, source: Union[str, Path, List[Union[str, Path]]]
    ) -> List[Document]:
        """Load documents from various sources."""
        source_path = Path(source) if isinstance(source, str) else source

        if isinstance(source_path, list):
            return await self._load_multiple_sources(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        try:
            if source_path.is_file():
                return await self._load_file(source_path)
            elif source_path.is_dir():
                return await self._load_directory(source_path)
            else:
                raise InvalidSourceError(f"Invalid source type: {source_path}")
        except Exception as e:
            self.metrics.add_error(str(e))
            raise

    async def process_documents(
        self, documents: List[Document]
    ) -> AsyncGenerator[ProcessedDocument, None]:
        """Process loaded documents with backpressure handling."""
        if not documents:
            return

        self._queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        async with self._processing_lock:
            try:
                # Set up the producer
                producer_done = asyncio.Event()

                # Start producer in background
                async def producer_task():
                    try:
                        await self._produce_documents(documents)
                    finally:
                        producer_done.set()
                        await self._queue.put(None)  # Signal completion

                producer = asyncio.create_task(producer_task())

                # Process documents
                try:
                    async with timeout(self.config.operation_timeout):
                        while not (producer_done.is_set() and self._queue.empty()):
                            doc = await self._queue.get()
                            if doc is None:
                                break

                            try:
                                processed = await self._process_single_document(doc)
                                if processed is not None:
                                    yield processed
                            finally:
                                self._queue.task_done()

                except asyncio.TimeoutError:
                    logger.error("processing_timeout")
                    self.cancel_processing()
                    raise TimeoutError("Document processing timed out")
                
                except Exception as e:
                    logger.exception("processing_error", error=str(e))
                    raise ProcessingError(f"Processing error: {e}")

                finally:
                    # Cleanup
                    if not producer.done():
                        producer.cancel()
                        try:
                            await producer
                        except asyncio.CancelledError:
                            pass

            except Exception as e:
                self.metrics.add_error(str(e))
                raise ProcessingError(f"Failed to process documents: {e}")

    async def _produce_documents(self, documents: List[Document]) -> None:
        """Producer that validates and queues documents."""
        try:
            for document in documents:
                if self._cancel_event.is_set():
                    break

                is_valid, error = await self._validate_document(document)
                if not is_valid:
                    logger.warning(
                        "document_validation_failed",
                        error=error,
                        document_id=document.metadata.get("source", "unknown"),
                    )
                    continue

                await self._queue.put(document)
        except Exception as e:
            logger.exception("producer_error", error=str(e))
            raise

    async def _process_single_document(
        self, document: Document
    ) -> Optional[ProcessedDocument]:
        """Process a single document through the pipeline."""
        try:
            # Apply preprocessors
            processed_doc = document
            for preprocessor in self.config.preprocessors:
                try:
                    async with timeout(self.config.operation_timeout):
                        processed_doc = await preprocessor.process(processed_doc)
                except asyncio.TimeoutError:
                    logger.warning(
                        "preprocessor_timeout",
                        document_id=document.metadata.get("source", "unknown"),
                    )
                    continue

            # Extract metadata
            metadata = await self._extract_metadata(processed_doc)

            # Chunk document
            chunks = await self._chunk_document(processed_doc)
            processed_chunks = []
            duplicates = 0

            # Process chunks with deduplication
            for chunk, chunk_meta in chunks:
                dedup_result = await self._deduplicate_chunk(chunk, chunk_meta)
                if dedup_result is not None:
                    processed_chunks.append(dedup_result)
                else:
                    duplicates += 1

            self.metrics.update_chunks(len(processed_chunks), duplicates)

            result = ProcessedDocument(
                chunks=processed_chunks,
                metadata=metadata,
                source=str(document.metadata.get("source", "unknown")),
            )

            self.metrics.update_processed_documents()
            return result

        except Exception as e:
            logger.exception(
                "document_processing_failed",
                error=str(e),
                document_id=document.metadata.get("source", "unknown"),
            )
            return None

    async def _load_file(self, path: Path) -> List[Document]:
        """Load single file with rate limiting."""
        if path.suffix[1:] not in self.config.allowed_extensions:
            raise InvalidSourceError(f"Unsupported file type: {path.suffix}")

        try:
            async with timeout(self.config.operation_timeout):
                # Apply rate limiting
                while not await self._rate_limiter.acquire():
                    await asyncio.sleep(1)

                loader = UnstructuredFileLoader(str(path) if isinstance(path, Path) else path.file)
                document = await asyncio.to_thread(loader.load)
                self.metrics.update_loaded_documents()
                return document

        except asyncio.TimeoutError:
            raise TimeoutError(f"Loading file {path} timed out")
        except Exception as e:
            raise ProcessingError(f"Failed to load file {path}: {e}")

    async def _load_directory(self, path: Path) -> List[Document]:
        """Load directory of files with progress bar."""
        try:
            async with timeout(self.config.operation_timeout):
                loader = DirectoryLoader(
                    str(path),
                    glob="**/*",
                    loader_cls=UnstructuredFileLoader,
                    use_multithreading=True, 
                    silent_errors=True,
                    show_progress=True,
                )
                documents = await asyncio.to_thread(loader.load)
                self.metrics.update_loaded_documents(len(documents))
                return documents

        except asyncio.TimeoutError:
            raise TimeoutError(f"Loading directory {path} timed out")
        except Exception as e:
            raise ProcessingError(f"Failed to load directory {path}: {e}")

    async def _load_multiple_sources(
        self, sources: List[Union[str, Path]]
    ) -> List[Document]:
        """Load multiple sources concurrently."""
        documents = []
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self.load_documents(source)) 
                for source in sources
            ]

        for task in tasks:
            try:
                docs = await task
                documents.extend(docs)
            except* Exception as e:
                self.metrics.add_error(str(e))
        
        self.metrics.update_loaded_documents(len(documents))
        return documents

    async def _validate_document(self, document: Document) -> Tuple[bool, Optional[str]]:
        """Run validation hooks on document."""
        for hook in self.config.validation_hooks:
            try:
                async with timeout(self.config.operation_timeout):
                    document = await hook(document)
            except asyncio.TimeoutError:
                return False, "Validation timeout"
            except Exception as e:
                return False, f"Validation failed: {str(e)}"
        return True, None

    async def _chunk_document(
        self, document: Document
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Split document into chunks with metadata."""
        try:
            async with timeout(self.config.operation_timeout):
                chunks = await asyncio.to_thread(
                    self._text_splitter.split_documents, [document]
                )

                return [
                    (
                        chunk.page_content,
                        ChunkMetadata(
                            id=str(uuid.uuid4()),
                            start=chunk.metadata.get("start", 0),
                            end=chunk.metadata.get("end", len(chunk.page_content)),
                            page=chunk.metadata.get("page", 1),
                            source=document.metadata.get("source", "unknown"),
                        ),
                    )
                    for chunk in chunks
                ]

        except asyncio.TimeoutError:
            raise TimeoutError("Document chunking timed out")
        except Exception as e:
            raise ChunkingError(f"Failed to chunk document: {e}")

    def _compute_chunk_hash(self, chunk: str) -> str:
        """Compute hash for chunk deduplication."""
        return hashlib.blake2b(chunk.encode(), digest_size=16).hexdigest()

    async def _deduplicate_chunk(
        self, chunk: str, metadata: ChunkMetadata
    ) -> Optional[Tuple[str, ChunkMetadata]]:
        """Deduplicate chunk if enabled."""
        if not self.config.enable_deduplication:
            return (chunk, metadata)

        chunk_hash = self._compute_chunk_hash(chunk)
        if chunk_hash in self._chunk_hashes:
            logger.debug("duplicate_chunk_found", hash=chunk_hash)
            return None

        self._chunk_hashes.add(chunk_hash)
        return (chunk, metadata)

    async def _extract_metadata(self, document: Document) -> DocumentMetadata:
        """Extract metadata from document."""
        try:
            async with timeout(self.config.operation_timeout):
                meta = document.metadata
                return DocumentMetadata(
                    id=str(uuid.uuid4()),
                    title=meta.get("title", ""),
                    author=meta.get("author", ""),
                    date=meta.get("date", ""),
                    source=meta.get("source", ""),
                    pages=meta.get("pages", 0),
                    language=meta.get("language", ""),
                    file_type=meta.get("file_type", ""),
                    creation_date=meta.get("creation_date", ""),
                    modification_date=meta.get("modification_date", ""),
                    size_bytes=meta.get("size_bytes", 0),
                )
        except asyncio.TimeoutError:
            logger.warning("metadata_extraction_timeout")
            return DocumentMetadata()
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            return DocumentMetadata()

    def cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        self._cancel_event.set()
        # Clear the queue immediately to prevent waiting on join()
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
        logger.info("processing_cancelled")

    def reset(self) -> None:
        """Reset processor state."""
        self._cancel_event.clear()
        self._chunk_hashes.clear()
        self.metrics = ProcessingMetrics()
        self._queue = None
        logger.info("processor_reset")

    async def close(self) -> None:
        """Clean up resources."""
        try:
            # Cancel processing first
            self.cancel_processing()
            
            # Wait a short time for queue to clear
            if self._queue is not None:
                try:
                    await asyncio.wait_for(self._queue.join(), timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("queue_cleanup_timeout")
        except Exception as e:
            logger.exception("cleanup_error", error=str(e))
        finally:
            self._chunk_hashes.clear()
            self._queue = None
            logger.info("processor_closed")
