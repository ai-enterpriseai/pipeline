import time
import asyncio
import hashlib
import uuid
import aiofiles
import aiofiles.os
from concurrent.futures import ThreadPoolExecutor

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
        # Handle Streamlit UploadedFile
        if str(type(source).__name__) == "UploadedFile":
            try:
                return await self._load_streamlit_file(source)
            except Exception as e:
                self.metrics.add_error(str(e))
                raise

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

    async def _load_streamlit_file(self, uploaded_file: Any) -> List[Document]:
        """Load single file from Streamlit upload with rate limiting."""
        file_suffix = Path(uploaded_file.name).suffix[1:]
        
        if file_suffix not in self.config.allowed_extensions:
            raise InvalidSourceError(f"Unsupported file type: {file_suffix}")
        
        content = await asyncio.to_thread(uploaded_file.read)
        content = content.decode('utf-8') if isinstance(content, bytes) else content

        import tempfile
        import os

        temp_file = None
        try:
            async with timeout(self.config.operation_timeout):
                # Apply rate limiting
                while not await self._rate_limiter.acquire():
                    await asyncio.sleep(1)

                # Create temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_suffix}')
                temp_file.write(content.encode('utf-8') if isinstance(content, str) else content)
                temp_file.flush()
                temp_file.close()  # Close file handle explicitly
                
                # Load document
                loader = UnstructuredFileLoader(temp_file.name)
                document = await asyncio.to_thread(loader.load)
                self.metrics.update_loaded_documents()
                return document

        except asyncio.TimeoutError:
            raise TimeoutError(f"Loading file {uploaded_file.name} timed out")
        except Exception as e:
            raise ProcessingError(f"Failed to load file {uploaded_file.name}: {e}")
        finally:
            # Clean up temp file
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")            

    async def _load_directory(self, path: Path) -> List[Document]:
        """Load directory of files asynchronously with concurrent processing."""
        try:
            async with timeout(self.config.operation_timeout):
                # Get all files asynchronously
                file_paths = await self._get_files_async(path)
                
                if not file_paths:
                    logger.warning(f"No supported files found in directory: {path}")
                    return []
                
                logger.info(f"Found {len(file_paths)} files to process in {path}")
                
                # Load files concurrently using TaskGroup
                documents = []
                max_concurrent = min(self.config.max_concurrent_files, len(file_paths))
                
                # Process files in batches to control concurrency
                for i in range(0, len(file_paths), max_concurrent):
                    batch = file_paths[i:i + max_concurrent]
                    
                    async with asyncio.TaskGroup() as group:
                        tasks = [
                            group.create_task(self._load_file_async(file_path))
                            for file_path in batch
                        ]
                    
                    # Collect results from completed tasks
                    for task in tasks:
                        try:
                            docs = await task
                            if docs:  # docs could be None or empty list if file failed to load
                                documents.extend(docs)
                        except Exception as e:
                            logger.warning(f"Failed to load file: {e}")
                            self.metrics.add_error(str(e))
                
                self.metrics.update_loaded_documents(len(documents))
                logger.info(f"Successfully loaded {len(documents)} documents from {len(file_paths)} files")
                return documents

        except asyncio.TimeoutError:
            raise TimeoutError(f"Loading directory {path} timed out")
        except Exception as e:
            raise ProcessingError(f"Failed to load directory {path}: {e}")

    async def _get_files_async(self, directory: Path) -> List[Path]:
        """Asynchronously get all supported files from directory."""
        files = []
        try:
            # Use asyncio for directory traversal
            async for file_path in self._walk_directory_async(directory):
                if file_path.is_file():
                    # Check file extension
                    extension = file_path.suffix[1:].lower()
                    if extension in self.config.allowed_extensions:
                        files.append(file_path)
                    else:
                        logger.debug(f"Skipping unsupported file: {file_path}")
        except Exception as e:
            logger.error(f"Error walking directory {directory}: {e}")
            raise ProcessingError(f"Failed to scan directory: {e}")
        
        return files

    async def _walk_directory_async(self, directory: Path):
        """Asynchronously walk through directory tree."""
        try:
            # Get directory contents asynchronously
            entries = await aiofiles.os.listdir(directory)
            
            for entry_name in entries:
                entry_path = directory / entry_name
                
                # Check if it's a file or directory asynchronously
                try:
                    stat_result = await aiofiles.os.stat(entry_path)
                    if stat_result.st_mode & 0o170000 == 0o040000:  # S_IFDIR
                        # It's a directory, recurse
                        async for sub_path in self._walk_directory_async(entry_path):
                            yield sub_path
                    else:
                        # It's a file
                        yield entry_path
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access {entry_path}: {e}")
                    continue
                        
        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot read directory {directory}: {e}")
            return

    async def _load_file_async(self, path: Path) -> Optional[List[Document]]:
        """Load a single file asynchronously with proper error handling."""
        try:
            # Apply rate limiting
            while not await self._rate_limiter.acquire():
                await asyncio.sleep(0.1)  # Shorter sleep for better concurrency
            
            # Load the file using UnstructuredFileLoader in thread pool
            loader = UnstructuredFileLoader(str(path))
            
            # Use a dedicated thread pool for CPU-bound operations
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                documents = await loop.run_in_executor(executor, loader.load)
            
            if documents:
                logger.debug(f"Successfully loaded {len(documents)} documents from {path}")
                return documents
            else:
                logger.warning(f"No content extracted from {path}")
                return []
                
        except Exception as e:
            logger.warning(f"Failed to load file {path}: {e}")
            self.metrics.add_error(f"File load error for {path}: {str(e)}")
            return None

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
