import asyncio
import torch 
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

from .embedder import (
    DenseEmbedder,
    SparseEmbedder
)
from .utils.configs import IndexerConfig 
from .utils.logging import setup_logger 
from .utils.types import (
    ProcessedDocument,
    IndexStats,
    VectorMetadata
)

logger = setup_logger(__name__)

class HealthStatus(BaseModel):
    """Health status of the index."""
    is_healthy: bool = Field(..., description="Indicates if the index is healthy.")
    last_check: datetime = Field(..., description="Timestamp of the last health check.")
    failed_checks: int = Field(..., description="Number of consecutive failed health checks.")
    errors: List[str] = Field(default_factory=list, description="List of error messages encountered.")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics as key-value pairs.")

class Indexer:
    def __init__(self, config: IndexerConfig, dense_embedder: DenseEmbedder, sparse_embedder: SparseEmbedder):
        """Initialize indexer with configuration."""
        self.config = config
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self._client = None  # Holds the client when initialized
        self.stats = IndexStats()
        self._upload_lock = asyncio.Lock()
        self._maintenance_task: Optional[asyncio.Task] = None
        self._health_status = HealthStatus(
            is_healthy=True,
            last_check=datetime.now(),
            failed_checks=0,
            errors=[],
            performance_metrics={}
        )

    @property
    def client(self):
        """Lazily initialize the QdrantClient if it hasn't been set."""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client

    @client.setter
    def client(self, value):
        """Allows setting the client, useful for testing."""
        self._client = value

    def _initialize_client(self) -> QdrantClient:
        """Initialize database client."""
        try:
            if self.config.use_local_db:
                # Initialize client for local database
                client = QdrantClient(
                    path=str(self.config.db_path),
                    timeout=self.config.operation_timeout
                )
            else:
                # Initialize client for cloud database
                client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.qdrant_api_key,
                    timeout=self.config.operation_timeout
                )
            logger.info("Successfully initialized Qdrant client")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise RuntimeError(f"Failed to initialize database client: {e}")

    async def start(self) -> None:
        """Start background tasks."""
        if self.config.maintenance.enable_optimization:
            self._maintenance_task = asyncio.create_task(
                self._run_maintenance_loop()
            )
        logger.info("Started background tasks")

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
        logger.info("Stopped background tasks")

    async def _run_maintenance_loop(self) -> None:
        """Run periodic maintenance tasks."""
        while True:
            try:
                # Wait for next maintenance interval
                await asyncio.sleep(
                    self.config.maintenance.optimization_interval * 3600
                )
                
                # Run optimization if enabled
                if self.config.maintenance.enable_optimization:
                    await self._optimize_index()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance task error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _optimize_index(self) -> None:
        """Optimize the vector index."""
        try:
            logger.info("Starting index optimization")
            
            # Get current index statistics
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.config.collection_name
            )
            
            # Check if optimization is needed
            if collection_info.points_count < 1000:
                logger.info("Index too small for optimization")
                return
            
            # Run optimization
            await asyncio.to_thread(
                self.client.optimize_index,
                collection_name=self.config.collection_name
            )
            
            logger.info("Index optimization completed")
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")

    async def initialize_collection(self) -> None:
        """Initialize or verify collection."""
        try:
            # Access the client property, initializing it if necessary
            collection_exists = self.client.collection_exists(
                self.config.collection_name
            )
            if not collection_exists:
                logger.info(f"Creating collection: {self.config.collection_name}")
                # Create collection with hybrid search configuration
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.config.dense_model_dimension,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams()
                    }
                )
                # Create payload index for metadata
                self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name="metadata.source",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.config.collection_name} exists")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise RuntimeError(f"Failed to initialize collection: {e}")

    async def delete_collection(self) -> None:
        """Delete the collection from the database."""
        try:
            await asyncio.to_thread(
                self.client.delete_collection,
                self.config.collection_name
            )
            logger.info(f"Successfully deleted collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise RuntimeError(f"Failed to delete collection: {e}")
    
    async def verify_index(self) -> Dict[str, Any]:
        """
        Basic index verification.
        
        Returns:
            Dictionary with verification results
        """
        try:
            # Verify collection exists and is accessible
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.config.collection_name
            )
            
            return {
                "status": "verified",
                "total_points": collection_info.points_count,
                "vectors_config": collection_info.config.params.vectors,
                "sparse_config": collection_info.config.params.sparse_vectors
            }
            
        except Exception as e:
            logger.error(f"Index verification failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _prepare_vectors(
        self,
        document: ProcessedDocument,
        dense_vectors: torch.Tensor,
        sparse_vectors: List[Dict[str, List]]
    ) -> List[models.PointStruct]:
        """Prepare vectors for database insertion."""
        try:
            points = []
            
            for i, (chunk, chunk_metadata) in enumerate(document.chunks):
                metadata = VectorMetadata(
                    document_id=document.metadata.id,
                    chunk_metadata=chunk_metadata,
                    source=document.source,
                    embedding_timestamp=datetime.now(),
                    chunk_text=chunk
                )

                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vectors[i],#.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vectors[i]["indices"],
                            values=sparse_vectors[i]["values"]
                        )
                    },
                    payload=metadata.model_dump()
                )
                
                points.append(point)
                
            return points
            
        except Exception as e:
            logger.error(f"Failed to prepare vectors: {e}")
            raise RuntimeError(f"Failed to prepare vectors: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def upload_batch(
        self,
        points: List[models.PointStruct],
        show_progress: bool = True
    ) -> None:
        """Upload a batch of vectors to the database."""
        try:
            async with self._upload_lock:
                # Define the sub-batch size for finer-grained progress control
                batch_size = self.config.batch_size
                total_uploaded = 0  # Track successfully uploaded points

                if show_progress:
                    with tqdm(total=len(points), desc="Uploading vectors") as pbar:
                        for i in range(0, len(points), batch_size):
                            batch = points[i:i + batch_size]
                            await asyncio.to_thread(
                                self.client.upload_points,
                                collection_name=self.config.collection_name,
                                points=batch,
                                parallel=self.config.parallel_uploads
                            )
                            pbar.update(len(batch))
                            total_uploaded += len(batch)  # Only count successfully uploaded batches
                else:
                    # Process entire batch if progress display is not needed
                    await asyncio.to_thread(
                        self.client.upload_points,
                        collection_name=self.config.collection_name,
                        points=points,
                        parallel=self.config.parallel_uploads
                    )
                    total_uploaded = len(points)  # All points uploaded if no progress bar

                # Update statistics with the successfully uploaded points
                self.stats.update_uploads(total_uploaded)
                logger.info(f"Successfully uploaded batch of {total_uploaded} vectors")

        except Exception as e:
            logger.error(f"Failed to upload batch: {e}")
            raise RuntimeError(f"Failed to upload batch due to {type(e).__name__}: {e}") from e

    async def index_documents(
        self,
        documents: List[ProcessedDocument],
    ) -> None:
        """Index documents with their dense and sparse vectors."""
        try:
            # Initialize collection if needed
            await self.initialize_collection()
            
            total_chunks = sum(len(doc.chunks) for doc in documents)
            batch = []
            
            for doc in documents:
                doc_chunks = [chunk[0] for chunk in doc.chunks]
                # Generate embeddings for the current document's chunks
                doc_dense = await self.dense_embedder.embed(doc_chunks)
                doc_sparse = await self.sparse_embedder.embed(doc_chunks)
                
                # logger.info(doc_dense)

                points = await self._prepare_vectors(doc, doc_dense, doc_sparse)
                batch.extend(points)
                
                if len(batch) >= self.config.batch_size:
                    await self.upload_batch(batch, show_progress=False)
                    batch = []
                                    
            # Upload remaining points
            if batch:
                await self.upload_batch(batch, show_progress=True)
                
            logger.info(f"Successfully indexed {total_chunks} chunks from {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to index documents:  {e}")
            raise RuntimeError(f"Failed to index documents: {e}")

    # TODO total_documents and total_chunks are not being stored, do later 
    async def get_stats(self) -> Dict[str, Any]:
        """Get current index statistics."""
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.config.collection_name
            )
            
            return {
                "vectors_count": collection_info.points_count,
                "indexed_documents": self.stats.total_documents,
                "indexed_chunks": self.stats.total_chunks,
                "last_update": self.stats.last_update.isoformat()
                if self.stats.last_update else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    async def close(self) -> None:
        """Clean up resources."""
        try:
            await self.stop()  # Stop background tasks
            await asyncio.to_thread(self.client.close)
            logger.info("Successfully closed database connection")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")